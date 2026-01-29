import pandas as pd
import json
import os
import glob
from dagster import asset, AssetExecutionContext
from ..resources import PostgreSQLResource, MongoDBResource, DataStorageResource
from .transformation import training_dataset
from .loading import trained_model, model_metrics


@asset(
    description="Store processed training data in PostgreSQL",
    group_name="storage",
    deps=[training_dataset],
)
def postgres_training_data(
    context: AssetExecutionContext,
    postgres: PostgreSQLResource,
    training_dataset: pd.DataFrame
):
    context.log.info("Storing training data in PostgreSQL")
    
    conn = postgres.get_connection()
    cursor = conn.cursor()
    
    try:
        # Drop and create table
        cursor.execute("""
            DROP TABLE IF EXISTS processed_data;
            CREATE TABLE processed_data (
                id SERIAL PRIMARY KEY,
                close_price NUMERIC(15, 6),
                high_price NUMERIC(15, 6),
                low_price NUMERIC(15, 6),
                open_price NUMERIC(15, 6),
                volume BIGINT,
                trade_date DATE,
                ticker VARCHAR(10),
                sma_50 NUMERIC(15, 6),
                trend VARCHAR(20),
                target INTEGER,
                price_change NUMERIC(15, 8),
                distance_from_sma NUMERIC(15, 8),
                momentum_5d NUMERIC(15, 8),
                volatility NUMERIC(15, 8),
                next_day_target INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX idx_ticker ON processed_data(ticker);
            CREATE INDEX idx_date ON processed_data(trade_date);
            CREATE INDEX idx_ticker_date ON processed_data(ticker, trade_date);
        """)
        
        # Insert data
        for _, row in training_dataset.iterrows():
            cursor.execute("""
                INSERT INTO processed_data (
                    close_price, high_price, low_price, open_price, volume,
                    trade_date, ticker, sma_50, trend, target,
                    price_change, distance_from_sma, momentum_5d, volatility, next_day_target
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                float(row.get('Close', 0)),
                float(row.get('High', 0)),
                float(row.get('Low', 0)),
                float(row.get('Open', 0)),
                int(row.get('Volume', 0)),
                row.get('Date'),
                row.get('Ticker'),
                float(row.get('SMA_50', 0)),
                row.get('Trend'),
                int(row.get('Target', 0)),
                float(row.get('Price_Change', 0)),
                float(row.get('Distance_from_SMA', 0)),
                float(row.get('Momentum_5d', 0)),
                float(row.get('Volatility', 0)),
                int(row.get('Next_Day_Target', 0)) if pd.notna(row.get('Next_Day_Target')) else None
            ))
        
        conn.commit()
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM processed_data")
        row_count = cursor.fetchone()[0]
        
        context.log.info(f"Successfully stored {row_count} rows in PostgreSQL")
        
        return {"rows_inserted": row_count, "table": "processed_data"}
        
    except Exception as e:
        conn.rollback()
        context.log.error(f"Error storing in PostgreSQL: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


@asset(
    description="Store model results and predictions in MongoDB",
    group_name="storage",
    deps=[trained_model, model_metrics],
)
def mongodb_model_results(
    context: AssetExecutionContext,
    mongodb: MongoDBResource,
    trained_model: dict,
    model_metrics: dict
):
    context.log.info("Storing model results in MongoDB")
    
    try:
        collection = mongodb.get_collection("model_results")
        
        result_doc = {
            "model_type": "RandomForestClassifier",
            "training_date": pd.Timestamp.now().isoformat(),
            "metrics": model_metrics,
            "configuration": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "features_used": [
                "SMA_50",
                "Price_Change",
                "Distance_from_SMA",
                "Momentum_5d",
                "Volatility"
            ]
        }
        
        result = collection.insert_one(result_doc)
        
        context.log.info(f"Stored model results in MongoDB with ID: {result.inserted_id}")
        
        return {
            "document_id": str(result.inserted_id),
            "collection": "model_results"
        }
        
    except Exception as e:
        context.log.error(f"Error storing in MongoDB: {e}")
        raise


@asset(
    description="Store raw stock data in MongoDB for flexible querying",
    group_name="storage",
    deps=["combined_raw_data"],
)
def mongodb_raw_data(
    context: AssetExecutionContext,
    mongodb: MongoDBResource,
    combined_raw_data: pd.DataFrame
):
    context.log.info("Storing raw data in MongoDB")
    
    try:
        collection = mongodb.get_collection("raw_stock_data_aapl")
        
        collection.delete_many({})
        
        documents = []
        for _, row in combined_raw_data.iterrows():
            doc = {
                "ticker": row.get('Ticker'),
                "date": row.get('Date').isoformat() if pd.notna(row.get('Date')) else None,
                "ohlcv": {
                    "open": float(row.get('Open', 0)) if pd.notna(row.get('Open')) else None,
                    "high": float(row.get('High', 0)) if pd.notna(row.get('High')) else None,
                    "low": float(row.get('Low', 0)) if pd.notna(row.get('Low')) else None,
                    "close": float(row.get('Close', 0)) if pd.notna(row.get('Close')) else None,
                    "volume": int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None
                },
                "ingested_at": pd.Timestamp.now().isoformat()
            }
            documents.append(doc)

        result = collection.insert_many(documents)
        
        collection.create_index([("ticker", 1), ("date", -1)])
        collection.create_index("date")
        
        context.log.info(f"Stored {len(result.inserted_ids)} documents in MongoDB")
        
        return {
            "documents_inserted": len(result.inserted_ids),
            "collection": "raw_stock_data_aapl"
        }
        
    except Exception as e:
        context.log.error(f"Error storing in MongoDB: {e}")
        raise


@asset(
    description="Bulk load raw stock data CSVs into separate PostgreSQL tables",
    group_name="storage",
    deps=["extract_stock_data"],  # Depends on the extraction asset
)
def postgres_raw_data(
    context: AssetExecutionContext,
    postgres: PostgreSQLResource,
    storage: DataStorageResource
):
    context.log.info("Starting bulk load of raw CSVs into PostgreSQL")
    
    raw_dir = storage.raw_dir
    csv_files = glob.glob(os.path.join(raw_dir, "stock_data_*.csv"))
    
    if not csv_files:
        context.log.warning(f"No stock_data_*.csv files found in {raw_dir}")
        return {"tables_created": 0}
        
    conn = postgres.get_connection()
    cursor = conn.cursor()
    
    tables_created = []
    
    try:
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            table_name = os.path.splitext(filename)[0]
            table_name = table_name.replace("-", "_").replace(" ", "_").lower()
            
            context.log.info(f"Processing {filename} -> Table {table_name}")
            
            cursor.execute(f"""
                DROP TABLE IF EXISTS {table_name};
                CREATE TABLE {table_name} (
                    trade_date DATE PRIMARY KEY,
                    close_price NUMERIC(15, 6),
                    high_price NUMERIC(15, 6),
                    low_price NUMERIC(15, 6),
                    open_price NUMERIC(15, 6),
                    volume BIGINT
                );
            """)
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                # Skipping first 3 lines
                for _ in range(3):
                    next(f)
                
                # Copy remaining data
                # Using columns list to match CSV order: Date, Close, High, Low, Open, Volume
                cursor.copy_expert(
                    f"COPY {table_name} (trade_date, close_price, high_price, low_price, open_price, volume) FROM STDIN WITH CSV",
                    f
                )
            
            # Verify count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            tables_created.append({"table": table_name, "rows": count})
            context.log.info(f"Loaded {count} rows into {table_name}")
            
        conn.commit()
        
        return {
            "tables_created": len(tables_created),
            "details": tables_created
        }
        
    except Exception as e:
        conn.rollback()
        context.log.error(f"Error loading Postgres raw data: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
