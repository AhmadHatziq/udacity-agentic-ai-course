import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import sys 
import json 

from sqlalchemy.sql import text, bindparam
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from openai import OpenAI

# Load OpenAI model 
dotenv.load_dotenv(dotenv_path='.env')
openai_api_key = os.getenv('UDACITY_OPENAI_API_KEY')
model = OpenAIServerModel(
    model_id='gpt-4o-mini',
    api_base='https://openai.vocareum.com/v1',
    api_key=openai_api_key,
)
client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=openai_api_key)

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "A3 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# List of all items 
paper_supplies_item_names = [x["item_name"] for x in paper_supplies]

# Pydantic models for order parsing and validation
ItemName = Literal[
    *paper_supplies_item_names 
]

class LineItem(BaseModel):
    item_name: ItemName
    qty: int = Field(gt=0)

class ParsedOrder(BaseModel):
    requested_by: Optional[str] = Field(
        None, description="YYYY-MM-DD or null, represents date where customer wants order by ie deadline"
    )
    requested_on: Optional[str] = Field(
        None, description="YYYY-MM-DD or null, represents request date where customer request is submitted"
    )
    line_items: List[LineItem]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate 'inventory' table and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, coverage=1.0, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)
        # print("=== TRANSACTIONS TABLE UPDATED =====")
        # print("=== INSERT ROW: ", str(transaction))

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific single item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_stock_level_multiple_items(
    item_names: List[str],
    as_of_date: Union[str, datetime]
) -> pd.DataFrame:
    """
    Retrieve the stock level of multiple inventory items as of a specified date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (List[str]): A list of item names to retrieve stock levels for. 
            Each item name must exactly match the corresponding `item_name` stored in the transactions table.
            E.g.: ["A4 paper", "Letter-sized paper", "Cardstock"]
        as_of_date (str, datetime, optional): The cutoff date (inclusive) for calculating stock. Defaults to current datetime.now() if None

    Returns:
        pd.DataFrame: A DataFrame containing one row per requested item with the following columns:
        - item_name (str): The inventory item name
        - current_stock (float): Net stock quantity as of the cutoff date

    Example:
        >>> item_names = ["A4 paper", "Rolls of banner paper (36-inch width)"]
        >>> get_stock_level_multiple_items(item_names)
            item_name                                  current_stock 
        0   A4 paper                                      272.0       
        1   Rolls of banner paper (36-inch width)         595.0       
    """
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    if as_of_date is None:
        as_of_date = datetime.now()

    if not item_names:
        return pd.DataFrame(columns=["item_name", "current_stock"])
    
    stock_query = text("""
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name IN :item_names
        AND transaction_date <= :as_of_date
        GROUP BY item_name
    """).bindparams(bindparam("item_names", expanding=True))

    return pd.read_sql(
        stock_query,
        db_engine,
        params={
            "item_names": item_names,
            "as_of_date": as_of_date,
        },
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.
    Logic: If there is a customer order and inventory levels are not enough, check if supplier can meet timeline. 
    - If not, forego order. 
    - If can meet, take the order. 

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11-100 units: 1 day
        - 101-1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0

def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }

def _row_to_doc(row: Dict) -> str:
    """
    Combine the joined row into one searchable 'document' string.
    Used in searching quote historu via TF IDF similarity. 
    """
    parts = [
        f"original_request: {row.get('original_request', '')}",
        f"quote_explanation: {row.get('quote_explanation', '')}",
        f"job_type: {row.get('job_type', '')}",
        f"order_size: {row.get('order_size', '')}",
        f"event_type: {row.get('event_type', '')}",
        f"total_amount: {row.get('total_amount', '')}",
        f"order_date: {row.get('order_date', '')}",
    ]

    # Lowercasing helps TF-IDF treat case-insensitive matches similarly
    return " | ".join(parts).lower()

def search_quote_history_tfidf(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.
    Uses TF-IDF to find similar quotes.

    The function searches both the original customer request and
    the explanation for the quote (from `quotes`) using vector similarity.
    Results are ranked by cosine similarity and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
            - similarity_score

    Example:
        >>> search_terms = ["school", "colorful", "urgent"]
        >>> search_quote_history_tfidf(search_terms)
        Search results:
            [{
                'original_request': 'I need to order 200 sheets of colorful cardstock, '
                                    '100 packs of paper plates, and 100 cups. '
                                    'I need these supplies delivered by April 10, 2025, '
                                    'for the party.',
                'total_amount': 58,
                'quote_explanation': 'Thank you for your order! For your upcoming party, '
                                     'I have calculated the total cost based on the '
                                     'requested items...',
                'job_type': 'school principal',
                'order_size': 'small',
                'event_type': 'party',
                'order_date': '2025-01-01T00:00:00',
                'similarity_score': 0.14573006855581366
            }, ...]

    Internal Steps:
        1) SELECT joined rows from `quotes` and `quote_requests`
        2) Combine each row into a single text document
        3) TF-IDF vectorize documents and the query
        4) Rank documents using cosine similarity and return top-k
    """
    query_text = " ".join(t.strip() for t in search_terms if t and t.strip()).lower()
    if not query_text:
        return []

    # Get all quotes and quote requests as a single table 
    sql = text("""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        ORDER BY q.order_date DESC
    """)

    with db_engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(sql)]

    if not rows:
        return []

    # Convert each row to a single document string 
    docs = [_row_to_doc(r) for r in rows]

    # Fit TF-IDF on the documents
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),     # unigrams + bigrams
        stop_words="english"    # remove common English stopwords (optional)
    )
    doc_matrix = vectorizer.fit_transform(docs)

    # Vectorize the query
    query_vec = vectorizer.transform([query_text])

    # Cosine similarity between query and all docs
    scores = cosine_similarity(query_vec, doc_matrix).ravel()

    # Sort indices by descending score
    ranked_idx = scores.argsort()[::-1]

    results: List[Dict] = []
    for i in ranked_idx[:limit]:
        r = rows[int(i)].copy()
        r["similarity_score"] = float(scores[int(i)])
        results.append(r)

    return results

def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]
    
def print_table_contents(table_name: str, db_engine: Engine):
    """
    Utility function to print the contents of a specified database table.

    Args:
        table_name (str): The name of the table to print.
        db_engine (Engine): The SQLAlchemy engine connected to the database.
    """
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", db_engine)
        print(f"Contents of table '{table_name}':")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Error printing table '{table_name}': {e}")

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""
############# Agent tool functions and helpers #####################
def extract_items_from_request(request_text: str) -> ParsedOrder:
    """
    Parse a free-form customer request into a structured purchase order using an LLM with schema-enforced (structured) output.

    This is responsible for translating natural-language customer requests (eg emails or chat messages) into a deterministic, machine-readable representation

    The LLM is constrained via a structured response schema (`ParsedOrder`) to ensure:
    - Only inventory items from the predefined catalog (`paper_supplies_item_names`) are returned
    - Quantities are extracted as integers representing raw sheet counts
    - Dates are normalized to ISO format (YYYY-MM-DD) or omitted if not specified
    - No hallucinated or unsupported items are introduced

    Args:
        request_text (str):
            The raw customer request containing item descriptions, quantities,
            and optional delivery timelines.

    Returns:
        ParsedOrder:
            A structured representation of the customer request containing:
            - requested_by (Optional[str]): Requested delivery date in YYYY-MM-DD format
            - requested_on (Optional[str]): The date that customer made the request, in YYYY-MM-DD format
            - line_items (List[LineItem]): Canonical inventory items and quantities

    Raises:
        ValidationError:
            If the LLM output does not conform to the `ParsedOrder` schema,
            ensuring downstream agents only receive valid structured data.
    """
    system_prompt = (
        "From the user content, find the most relevant items, date and counts."
        "Use raw sheet counts. If 'ream' is used, 1 ream contains 500 sheets. "
        f"Only use the provided item list: {str(paper_supplies_item_names)}"
    )
    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "messages": [{"role": "system", "content": system_prompt},
            {"role": "user", "content": request_text}], 
        "response_format": ParsedOrder
    }   
    response = client.beta.chat.completions.parse(**payload)
    choice = response.choices[0].message 
    parsed: ParsedOrder = choice.parsed
    print(f"Function: extract_items_from_request. Input: {request_text}\nOutput: {parsed}")
    return parsed
 
def get_unit_price(item_names: List[str]) -> Dict[str, float]:
    """
    Function used to extract the unit prices for a list of items from the inventory table.

    Args:
        item_names (List[str]): The list of item names to get unit prices for.

    Returns:
        Dict: A dictionary mapping item names to their unit prices.

    Example:
        >>> tool_quote_get_unit_price(["A4 Paper", "A3 Paper"])
        >>> {"A4 Paper": 0.05, "A3 Paper": 0.10}
    """
    price_dict = {}
    for item_name in item_names: 
        for paper_supply_item in paper_supplies: 
            if paper_supply_item["item_name"].lower() == item_name.lower(): 
                price_dict[item_name] = paper_supply_item["unit_price"]
    return price_dict

@tool 
def tool_inventory_compare_current_stock_with_customer_requirements(current_stock: str, customer_requirements: str, 
                                                                    minimum_stock_level: str) -> List[Dict]: 
    """
    Uses a LLM to compare current inventory levels with customer quantity needs and min stock levels. 
    Args:
        current_stock (str): The current inventory levels
        customer_requirements (str): The quantity needs of the customer 
        minimum_stock_level (str): The minimum stock levels of items 

    Returns:
        List[Dict]: list of items, with each item having attribute: 
            item_name, current_stock, customer_needs, min_stock_level, amount_to_order, current_stock_sufficient
    """
    system_prompt = (
        "Your role is to compare item levels to meet customer and inventory requirements."
        "You will have the current stock values, minimum stock level required and customer needs."
        "Calculate the 'amount_to_order' value, which is the difference we need to meet customer needs and the min stock level required."
        "Return a dictionary representing if the item inventory level can meet customer requirements (current_stock_sufficient: boolean), the relevant counts etc"
        f"Eg {{item_name: A4 paper, current_stock: int, customer_needs: int, min_stock_level: int, amount_to_order: int,  current_stock_sufficient: Boolean }}"
    )
    user_prompt = json.dumps(
    {   "Current_requested_item_inventory_levels": current_stock, 
        "Item_minimum_stock_levels": minimum_stock_level, 
        "Customer_requirements": customer_requirements
    }, ensure_ascii=False
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ], temperature=0
    )
    return response.choices[0].message.content

@tool 
def tool_inventory_get_min_stock_level(item_names: List[str]) -> Dict: 
    """
    Uses a LLM to extract the min stock levels relevant to the item names. 
    Args:
        item_names: List[str]: The items to obtain the min stock level. 

    Returns:
        Dict: The min stock level for each item 
    """

    # Query inventory table 
    df = pd.read_sql(f"SELECT * FROM inventory", db_engine)

    system_prompt = (
        "Your role is to extract the minimum stock levels of items from a database table."
        "Extract minimum stock levels for the requested items.\n"
        "Return dict list ONLY as a list of objects: "
        '[{"item_name": "...", "min_stock_level": 0}, ...].\n'
        "If an item is not present in the table, include it with min_stock_level = 0."
    )

    user_prompt = json.dumps({
        "Items_to_extract_for": item_names, 
        "Inventory_table": df.to_string(index=False)
    }, ensure_ascii=False)

    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
    )
    return response.choices[0].message.content

@tool
def tool_inventory_get_all_inventory(as_of_date: Union[str, datetime]) -> Dict[str, int]:
    """
    Tool used to etrieve the stock level of a specific single item as of a given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating inventory stock.

    Returns:
        Dict[str, int]: Stock levels per item as of the given date
    """
    return get_all_inventory(as_of_date)

@tool
def tool_inventory_get_stock_single_item(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Tool used to etrieve the stock level of a specific single item as of a given date.

    Args:
        item_name (str): The name of the single item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.\
        
    Example:
        >>> item_name = "A4 paper"
        >>> tool_inventory_get_stock_single_item(item_names)
            item_name                                  current_stock
        0   A4 paper                                      272.0
    """
    return get_stock_level(item_name, as_of_date)

@tool 
def tool_inventory_get_stock_multiple_items(item_names: List[str], as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Tool used to retrieve the stock level of multiple items as of a given date.

    Args:
        item_names (List[str]): A list of item names to retrieve stock levels for. 
            Each item name must exactly match the corresponding `item_name` stored in the transactions table.
            E.g.: ["A4 paper", "Letter-sized paper", "Cardstock"]
        as_of_date (str, datetime, optional): The cutoff date (inclusive) for calculating stock. Defaults to current datetime.now() if None

    Returns:
        pd.DataFrame: A DataFrame containing one row per requested item with the following columns:
        - item_name (str): The inventory item name
        - current_stock (float): Net stock quantity as of the cutoff date

    Example:
        >>> item_names = ["A4 paper", "Rolls of banner paper (36-inch width)"]
        >>> tool_inventory_get_stock_multiple_items(item_names)
            item_name                                  current_stock
        0   A4 paper                                      272.0
        1   Rolls of banner paper (36-inch width)         595.0
    """
    return get_stock_level_multiple_items(item_names, as_of_date)

@tool 
def tool_inventory_get_supplier_delivery_date(request_date: str, quantity_to_order: int) -> str: 
    """
    Tool used to get the supplier delivery date for a single item that need to be ordered. 

    Args:
        request_date (str, datetime): The datetime of when the request is submitted. Will be the request date. 
            Will be formatted with the command: datetime.fromisoformat(request_date.split("T")[0])
        quantity_to_order (int): The amount of items to order. 

    Returns:
        str: formatted delivery date as a string 

    Example:
        >>> tool_inventory_get_supplier_delivery_date("04/02/2025", 15)
        >>> "25/02/2025"
            Item requires several days to arrive at the specified returned date
    """
    return get_supplier_delivery_date(request_date, quantity_to_order)

@tool 
def tool_inventory_can_we_meet_customer_requirements(
    item_name: str, amount_to_order: int, customer_deadline: str, supplier_delivery_date: str,) -> bool: 
    """
    Tool used to determine for a single item, if we can meet customer requirements based on supplier delivery date, current stock levels. 

    Args:
        item_name (str): The name of the item. 
        amount_to_order (int): The amount of items to order. 
        customer_deadline (str): The customer deadline date in ISO format (YYYY-MM-DD).
        supplier_delivery_date (str): The supplier delivery date for the item in ISO format (YYYY-MM-DD)

    Returns:
        bool: True if we can meet customer requirements, False otherwise.

    Example:
        >>> tool_inventory_can_we_meet_customer_requirements("A4 Paper", 15, customer_deadline="21/02/2025", supplier_delivery_date="28/02/2025")
        >>> False as the supplier delivery date is after customer deadline.

        >>> tool_inventory_can_we_meet_customer_requirements("A4 Paper", 15, customer_deadline="21/02/2025", supplier_delivery_date="15/02/2025")
        >>> True as the supplier delivery date is before customer deadline.

        >>> tool_inventory_can_we_meet_customer_requirements("A4 Paper", 0, customer_deadline, supplier_delivery_date)
        >>> True there is no need to order anything as amount_to_order is 0.
    """
    if amount_to_order <= 0:
        return True  # No need to order anything, so we can meet requirements
    
    try: 
        deadline_date = datetime.fromisoformat(customer_deadline)
        delivery_date = datetime.fromisoformat(supplier_delivery_date)
        return delivery_date <= deadline_date
    except (ValueError, TypeError):
        print(f"WARN (tool_inventory_can_we_meet_customer_requirements): Invalid date format(s) for item {item_name}, assuming cannot meet requirements.")
        return False  # Invalid date format, assume we cannot meet requirements

@tool 
def tool_quote_search_history_sql(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Tool used to search historical quotes based on search terms using SQL LIKE.

    Args:
        search_terms (List[str]): The list of keywords to search for. Keywords are derived from customer request. 
        limit (int, optional): The maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    return search_quote_history(search_terms, limit=limit)

@tool 
def tool_quote_search_history_vector(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Tool used to search historical quotes based on search terms using TF-IDF vector similarity.

    Args:
        search_terms (List[str]): The list of keywords to search for. Keywords are derived from customer request. 
        limit (int, optional): The maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
            - similarity_score

    Example:
        >>> search_terms = ["school", "colorful", "urgent"]
        >>> tool_quote_search_history_vector(search_terms)
        Search results:
            [{
                'original_request': 'I need to order 200 sheets of colorful cardstock, '
                                    '100 packs of paper plates, and 100 cups. '
                                    'I need these supplies delivered by April 10, 2025, '
                                    'for the party.',
                'total_amount': 58,
                'quote_explanation': 'Thank you for your order! For your upcoming party, '
                                     'I have calculated the total cost based on the '
                                     'requested items...',
                'job_type': 'school principal',
                'order_size': 'small',
                'event_type': 'party',
                'order_date': '2025-01-01T00:00:00',
                'similarity_score': 0.14573006855581366
            }, ...]
    """
    return search_quote_history_tfidf(search_terms, limit=limit)

@tool 
def tool_quote_get_discount_thresholds(item_names: List[str], historical_quotes: List[Dict]) -> List[Dict[str, str]]: 
    """
    Uses a LLM to extract the discount thresholds for items based on historical quotes.
    Args:
        item_names (List[str]): List of item names. 
        historical_quotes (List[Dict]): A list of historical quotes. 

    Returns:
        List[Dict]: list of items, with each item having attribute: 
            item_name (str), discount_threshold (str), unit_discount_price (str), reasoning (str)
    """
    system_prompt = (
        "Your role is to analyze historical quotes and infer the discount theresholds."
        "You will be given the item names and a text dump of previous historical quotes."
        "If there is no discount threshold for an item, set discount_threshold to 'NA' and unit_discount_price to 'NA'."
        "Use raw sheet counts. If 'ream' is used, 1 ream contains 500 sheets"
        "Return a JSON array of objects, one per item with the below fields and reasoning.\n\n"
        f"Eg {{item_name: str, discount_threshold: str, unit_discount_price: str, reasoning: str }}"
        "Formatting rules:\n"
        "- discount_threshold MUST be a string\n"
        "   - If numeric, represent it as a number encoded as a string (e.g. \"500\", \"10000\")\n"
        "   - If unknown or not applicable, use \"NA\"\n"
        "- unit_discount_price MUST be a string\n"
        "   - If numeric, use decimal format as string (e.g. \"0.045\")\n"
        "   - If unknown, use \"NA\"\n\n"
    )
    user_prompt = json.dumps({
        "Items" : item_names, 
        "Historical quotes": historical_quotes
    }, ensure_ascii=False)

    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ], temperature=0
    )
    return response.choices[0].message.content

@tool 
def tool_quote_apply_discount(original_unit_prices: List[Dict], quantity_ordered: List[Dict], 
                              discount_thresholds: List[Dict]) -> List[Dict]: 
    """
    Apply discounts to original unit prices based on quantity ordered and historical discount thresholds.

    Args:
        original_unit_prices (List[Dict): List of original unit prices per item. Eg [{"A4 Paper": 0.05}, {"A3 Paper": 0.10}]
        quantity_ordered (List[Dict]): Customer order quantity. Eg [{'item_name': 'Cups', 'qty': 5218}, ...]
        discount_thresholds (List[Dict]): List of discount thresholds per item 
            Eg [{
        "item_name": "Flyers",
        "discount_threshold": "1000",
        "unit_discount_price": "0.15",
        "reasoning": "The quote for 1000 full-color flyers indicates a bulk pricing of $0.15 each, suggesting that the discount threshold is at least 1000 flyers."
        }, ...]

    Returns:
        List[Dict]: One dictionary per item with the following fields:
            - item_name (str)
            - quantity_ordered (int)
            - original_unit_price (float)
            - final_unit_price (float) - must be lower than original_unit_price if discount applied
            - discount_applied (bool)
            - historical_discount_threshold (str) - NA if none
            - historical_unit_discount_price (str) - NA if none
            - reasoning (str) 
    """

    system_prompt = (
    "Return ONLY valid JSON (no markdown, no comments, no trailing commas). "
    "You are a pricing assistant. Your job is to compute final_unit_price per item.\n\n"

    "INPUTS:\n"
    "1) original_unit_prices: list of dicts like [{\"A4 Paper\": 0.05}, {\"A3 Paper\": 0.10}]\n"
    "2) quantity_ordered: list of dicts like [{\"item_name\": \"Cups\", \"qty\": 5218}, ...]\n"
    "3) discount_thresholds: list of dicts containing raw text dumps of historical data"

    "NORMALIZATION RULES:\n"
    "- Treat discount_threshold and unit_discount_price as NUMBERS even if provided as strings.\n"
    "- Match items case-insensitively by item_name. original_unit_prices keys are item names.\n"

    "PRICING RULES (MUST FOLLOW):\n"
    "A) Never increase price. final_unit_price MUST be <= original_unit_price whenever original_unit_price is not null.\n"
    "B) Determine eligibility:\n"
    "   - If there is historical data for the item and qty >= discount_threshold => eligible.\n"
    "   - Otherwise not eligible.\n"
    "C) If eligible:\n"
    "   - Compute a candidate price from historical data: hist_price = unit_discount_price.\n"
    "   - IMPORTANT: hist_price is NOT automatically applied.\n"
    "   - Choose final_unit_price = min(original_unit_price * 0.98, hist_price, original_unit_price).\n"
    "     (This guarantees a discount but never raises the price.)\n"
    "   - If this min equals original_unit_price (i.e. no real discount possible), then set discount_applied=false.\n"
    "D) If not eligible:\n"
    "   - final_unit_price = original_unit_price, discount_applied = false.\n\n"

    "OUTPUT FORMAT:\n"
    "Return a JSON array. Each element must be:\n"
    "{\n"
    "  \"item_name\": string,\n"
    "  \"quantity_ordered\": integer,\n"
    "  \"original_unit_price\": number,\n"
    "  \"final_unit_price\": number or null,\n"
    "  \"discount_applied\": boolean,\n"
    "  \"historical_discount_threshold\": string or \"NA\",\n"
    "  \"historical_unit_discount_price\": string or \"NA\",\n"
    "  \"reasoning\": string\n"
    "}\n\n"

    "REASONING REQUIREMENTS:\n"
    "- If eligible, explain: qty vs threshold, original price, historical price, and why final price chosen.\n"
    "- If not eligible, explain why.\n"
    )
    user_prompt = json.dumps(
        {
            "original_unit_prices": original_unit_prices,
            "quantity_ordered": quantity_ordered,
            "discount_thresholds": discount_thresholds,
        },
        ensure_ascii=False
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ], temperature=0
    )
    return response.choices[0].message.content

@tool 
def tool_sales_confirm_restock(item_name: str, quantity: int, unit_price: float, date: Union[str, datetime]) -> str: 
    """
    This tool is used to confirm restock order of a single item. 

    Args:
        item_name (str): Name of item to restock.
        quantity (int): Quantity needed to restock. 
        unit_price (float): Unit price of item. 
        date (str): Transaction date as ISO date string (YYYY-MM-DD).

    Returns:
        str: Outcome of transaction.  
    """
    transaction_id = create_transaction(
        item_name=item_name, transaction_type='stock_orders', 
        quantity=quantity, price=unit_price*quantity, date=date
    )

    return f"Restock update success for item {item_name} with transaction ID: {transaction_id}"

@tool 
def tool_sales_confirm_sale(item_name: str, quantity: int, unit_price: float, date: Union[str, datetime]) -> str: 
    """
    This tool is used to confirm sale of a single item to the customer. 

    Args:
        item_name (str): Name of item in the sale order. 
        quantity (int): Quantity sold to customer. 
        unit_price (float): Sale unit price of the item. Either discounted unit price or original unit price. 
        date (str): Transaction date as ISO date string (YYYY-MM-DD).

    Returns:
        str: Outcome of transaction.  
    """
    transaction_id = create_transaction(
        item_name=item_name, transaction_type='sales', 
        quantity=quantity, price=unit_price*quantity, date=date
    )

    return f"Sale update success for item {item_name} with transaction ID: {transaction_id}"

@tool 
def tool_raise_alert(message: str, severity: Literal["low","medium","high","critical"], transaction_ids: str) -> None:
    """
    This tool is used to raise an alert for manual human intervention. 

    Args:
        message (str): The alert message to be raised.
        severity (Literal["low","medium","high","critical"]): The severity level of the alert: The severity level of the alert.
        transaction_ids (str): The related transaction IDs for context.

    Returns: 
        None
    """
    print(f"============ ALERT: {message} ===============")
    print(f"Severity Level: {severity}")
    print(f"Related Transaction IDs: {transaction_ids}")
    print("Simulating sending alert to operations team via slack/email/SMS...")
    print("Simulating logging alert to monitoring system eg Jira ticket...")
    return None

@tool 
def tool_sales_get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    This tool is used to get the cash balance of the system as of a given date. 

    Args:
        as_of_date (str or datetime): The cutoff ISO date (inclusive) for calculating cash balance.
    
    Returns:
        float: The cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    return get_cash_balance(as_of_date)

@tool 
def tool_sales_get_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    This tool is used to get the financial report of the system as of a given date. 
    Includes metrics such as inventory valuation, combined asset total etc. 

    Args:
        as_of_date (str or datetime): The cutoff ISO date (inclusive) for calculating financial report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    return generate_financial_report(as_of_date)

############# End of agent tool functions ##############

# Set up your agents and create an orchestration agent that will manage them.
############# Agent classes #####################
class SalesAgent(ToolCallingAgent):
    """After considering inventory levels and discount pricing, decide to proceed with the possible sales. Will mutate DB"""
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        super().__init__(
            tools=[tool_sales_confirm_restock, tool_sales_confirm_sale, 
                   tool_sales_get_cash_balance, tool_sales_get_financial_report, 
                   tool_raise_alert],
            model=model,
            name='SalesAgent',
            description="""You are an agent dealing with finalizing sales operations and customer responses. 
            """,)

class InventoryAgent(ToolCallingAgent):
    """Handles inventory checks. Does not mutate database."""
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        super().__init__(
            tools=[tool_inventory_get_stock_single_item, tool_inventory_get_stock_multiple_items, tool_inventory_get_all_inventory, 
                   tool_inventory_compare_current_stock_with_customer_requirements, tool_inventory_get_min_stock_level, 
                   tool_inventory_get_supplier_delivery_date, tool_inventory_can_we_meet_customer_requirements],
            model=model,
            name='InventoryAgent',
            description="""You are an agent dealing with inventory operations. 
            """,)

class QuotingAgent(ToolCallingAgent):
    """Compare customer request with similar historical quotes and decide to apply bulk discounts."""
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        super().__init__(
            tools=[tool_quote_search_history_sql, 
                   tool_quote_search_history_vector, 
                   tool_quote_get_discount_thresholds, 
                   tool_quote_apply_discount],
            model=model,
            name='QuoteAgent',
            description="""You are an agent responsible for deciding to offer discounts. 
            """,)

class Orchestrator(ToolCallingAgent):
    """Orchestrates the sales workflow."""
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        self.inventory_agent = InventoryAgent(model)
        self.quoting_agent = QuotingAgent(model)
        self.sales_agent = SalesAgent(model)

        @tool 
        def handle_customer_request(customer_request:str, date_of_request: Optional[str]=None, due_date: Optional[str]=None) -> Dict: 
            """
            End-to-end orchestration handler:
            1) Parse customer request to get relevant items, quantities and date
            2) Send to Inventory Agent 
                - Check if customer request can be fulfilled with current stock. 
                - If not, check if a re-order can meet customer deadline
            3) Send to Quote Agent 
                - Check to offer bulk discounts due to historical quotes 
                - Decides final item price 
            4) Send to Sales Agent 
                - Based on inventory report and quote report, decide which items to proceed with restock and sale operations. 
                - Determine financial health and raise alert for human intervention if needed.
                - Returns customer response. 

            Args:
                customer_request (str): The raw customer request containing item descriptions, quantities, optional delivery timelines etc.
                date_of_request (Optional[str]): ISO date string (YYYY-MM-DD) representing the request date ie date that customer made the request
                due_date (Optional[str]): ISO date string (YYYY-MM-DD) representing the customer deadline delivery date. 
                    
            Returns:
                Dict: A dictionary indicating the result of processing the order and feedback to the customer 
            """
            ## Data preprocessing: Parsing and fetching 
            # Extract items from customer text
            parsed_items = extract_items_from_request(customer_request + ". Date of Request: " + str(date_of_request) + ". Due date: " + str(due_date))
            print("Customer request: ", customer_request)
            print("Parsed items: ", parsed_items)
            deadline_date = parsed_items.requested_by
            request_date = parsed_items.requested_on
            extracted_item_counts = [item.model_dump() for item in parsed_items.line_items]
            extracted_items = [item.item_name for item in parsed_items.line_items]

            # Get original unit prices for the items 
            item_original_unit_prices = get_unit_price(extracted_items)

            # Get inventory report from inventory agent
            inventory_report = self.inventory_agent.run(
                f"""
                You are InventoryAgent. Your job is to produce a fulfillment feasibility report per requested item using ONLY the available tools.

                INPUTS
                - item_names: {extracted_items}
                - item_counts: {len(extracted_items)}
                - customer_requirements: {extracted_item_counts}   # list of {{"item_name": str, "qty": int}}
                - request_date (ISO): {request_date}              # when order is placed / stock snapshot date
                - customer_deadline (ISO): {deadline_date}        # when customer needs items delivered

                HARD OUTPUT RULES (IMPORTANT)
                - Final output MUST be a valid JSON array (not Markdown, not code fences).
                - Do NOT include any extra keys beyond the schema below.
                - Do NOT include explanations, “Observations:”, pipes like “|”, or any additional text.
                - All numbers must be integers (convert 764.0 -> 764).
                - Dates must be ISO strings "YYYY-MM-DD". If not applicable, use null.

                REQUIRED OUTPUT SCHEMA (one object per item, same order as customer_requirements)
                [
                    {{
                        "item_name": str,
                        "current_stock": int,
                        "customer_needs": int,
                        "min_stock_level": int,
                        "amount_to_order": int,
                        "supplier_delivery_date": str|null,
                        "can_we_meet_customer_requirements": bool
                    }}
                ]

                ALGORITHM (follow exactly)
                1) Get current stock for ALL items as of request_date:
                - If len(item_names) > 1 call tool_inventory_get_stock_multiple_items(item_names, request_date)
                - Else call tool_inventory_get_stock_single_item(item_name, request_date)
                - If len(item_names) > 10, call tool_inventory_get_all_inventory(request_date)

                2) Get minimum stock levels for the same item_names using tool_inventory_get_min_stock_level(item_names).

                3) Compare the current stock levels, customer quantity requirements and minimum stock level. 
                Use the tool tool_inventory_compare_current_stock_with_customer_requirements.
                Customer quantity requirements: {extracted_item_counts}. 
                The field "amount_to_order" will be used in the next step. 

                4) For each item where amount_to_order > 0:
                - Call tool_inventory_get_supplier_delivery_date(request_date, amount_to_order)
                - Store as supplier_delivery_date for that item
                For items with amount_to_order == 0, supplier_delivery_date = null.

                5) Combine all the findings in a list of dictionary for the items. Follow below format: 
                Eg format: [{{item_name, current_stock, customer_needs, min_stock_level, amount_to_order, supplier_deliver_date }}, ...]

                6) Iterate over all items with the tool tool_inventory_can_we_meet_customer_requirements. 
                The customer deadline due date is: {deadline_date}. 

                7) Combine all the findings in a list of dictionary for the items. Follow the OUTPUT format. 
                Return as answer. 
                """
            )
            print("Inventory Agent Report: ", inventory_report)

            """
            Sample inventory report response format. 
            can_we_meet_customer_requirements is true if amount_to_order is 0 or can be delivered before customer deadline.
            [
             {'item_name': 'Flyers', 'current_stock': 764, 'customer_needs': 5000, 'min_stock_level': 142, 
                'can_we_meet_customer_requirements': True, 'amount_to_order': 4236, 'supplier_deliver_date': '2025-04-24'}, 
             {'item_name': 'Invitation cards', 'current_stock': 256, 'customer_needs': 10000, 'min_stock_level': 57, 
                'can_we_meet_customer_requirements': True, 'amount_to_order': 9744, 'supplier_deliver_date': '2025-04-24'}, 
             {'item_name': 'Poster paper', 'current_stock': 493, 'customer_needs': 2000, 'min_stock_level': 64, 
                'can_we_meet_customer_requirements': True, 'amount_to_order': 1507, 'supplier_deliver_date': '2025-04-24'}
            ]
            """

            # Get quote report from quoting agent
            quote_report = self.quoting_agent.run(
                f"""
                The customer requested items are: {extracted_items}. 
                Customer order quantity requirements: {extracted_item_counts}.
                Item original unit prices: {item_original_unit_prices}

                1. Based on the customer request, generate at least 8 keywords that describe the customer request below. 
                Possible keyword categories could be urgency type (determine if urgent based on request and deadline date), 
                    job type, event type, order size, urgency, color requirements etc.
                Keyword restrictions: must be a single word, must not contain numbers or special characters, must be lowercase.
                Eg: keywords: ["school", "colorful", "urgent", "teacher", "office", "assembly", "concert", "parade", "party"]
                Customer request: {customer_request}
                Customer request date: {request_date}
                Customer deadline due date: {deadline_date}

                2. Using the keywords, search historical quotes using both the tool tool_quote_search_history_sql and tool_quote_search_history_vector.
                Combine the results from both tools. The historical quotes will be used in the next step.
                Format: historical_quotes = List[{{'original_request', 'total_amount', 'quote_explaination', 'job_type',... }}, ...]

                3. Using the historical quotes found, use the tool tool_quote_get_discount_thresholds to get the discount thresholds. 
                Format for argument historical_quotes = List[{{'original_request', 'total_amount', 'quote_explaination', 'job_type',... }}, ...]
                
                4. Use the tool tool_quote_apply_discount to apply discounts based on the original unit prices, customer quantity requirements 
                    and historical discount thresholds.
                Item original unit prices: {item_original_unit_prices}
                Customer order quantity requirements: {extracted_item_counts}.

                If there is a discount error, fall back to using original_unit_price. 
                Returns below as final answer:
                    List[Dict]: One dictionary per item with the following fields:
                        - item_name (str)
                        - quantity_ordered (int)
                        - original_unit_price (float)
                        - final_unit_price (float) - must be lower than original_unit_price if discount applied
                        - discount_applied (bool)
                        - historical_discount_threshold (str)
                        - historical_unit_discount_price (str)
                        - reasoning (str) 
                """
            )

            print("Inventory Agent Report: ", inventory_report)
            print("Quoting Agent Report: ", quote_report)

            # Get quote report from quoting agent
            sales_reply = self.sales_agent.run(
                f"""
                You are SalesAgent. Your job has TWO phases in strict order:

                PHASE A — TRANSACTIONS (must be completed before writing the customer message) - Update transactions DB and determine financial health
                PHASE B — CUSTOMER RESPONSE (final answer)

                You are given:
                - request_date: {request_date}
                - extracted_items: {extracted_items}
                - inventory_report: {inventory_report}
                - quote_report: {quote_report}
                - original_customer_request: {customer_request}

                Rules (non-negotiable):
                1) Process EACH unique item_name exactly once. Use item_name as the join key between inventory_report and quote_report.
                2) Only process items where inventory_report.can_we_meet_customer_requirements == True.
                3) For each processable item, do exactly two tool calls in this order:
                a) tool_sales_confirm_restock(item_name, quantity=amount_to_order, unit_price=original_unit_price, date=request_date)
                b) tool_sales_confirm_sale(item_name, quantity=customer_needs, unit_price=final_unit_price, date=request_date)
                4) Capture the sale transaction ID returned by tool_sales_confirm_sale for each item.
                5) If any item cannot be met (can_we_meet_customer_requirements == False), do NOT call tools for it. Mention it politely in the response.
                6) After processing all items, do exactly 2 tools calls to assess financial health:
                a) tool_sales_get_cash_balance(request_date) to get the cash balance as of request_date.
                b) tool_sales_get_financial_report(request_date) to get the financial report as of request_date.
                7) If cash balance < $1000 or inventory_value < $1000, raise an alert using tool_raise_alert with severity "high" and include all relevant transaction IDs. 
                    In the alert message, mention a summary of the details eg cash balance and inventory value. 
                8) After PHASE A is fully complete, proceed to PHASE B to write the customer response and return as final_answer.
                
                Pricing rules:
                - Use final_unit_price for the SALE cost.
                - Discounted item means quote_report.discount_applied == True.
                - Line total = quantity_ordered * final_unit_price.
                - Grand total = sum of all fulfilled line totals.
                - Never claim a discount if final_unit_price == original_unit_price.

                Delivery/timeline:
                - If inventory_report includes supplier_delivery_date, mention the latest (max) supplier_delivery_date among fulfilled items as "estimated ready/ship date".
                - Also explicitly acknowledge the customer's requested delivery deadline if present in the original request (e.g., “by May 15, 2025”) 
                  and state whether the supplier_delivery_date appears to meet it (simple comparison if both are dates; otherwise phrase cautiously).

                Output requirements:
                - Your FINAL ANSWER must be ONLY the customer-facing message (no JSON, no internal reasoning, no tool logs).
                - The message must include:
                (a) A short friendly opener referencing the customer context (infer event / purpose / occupation from original request).
                (b) A clear per-item summary for fulfilled items: item name, quantity, final unit price, and line total.
                (c) A short “Discounts applied” section listing discounted items (or “None”).
                (d) The grand total.
                (e) The customer initial deadline date and the "estimated ready/ship date".
                (f) Sale transaction IDs. 
                (g) Any unfulfilled items (if any) + what the customer can do next 
                    (eg adjust quantity, substitute item, accept later delivery, 
                    if customer has queries, can contact us with the transaction IDs and we will he happy to help).

                Write style:
                - Friendly, professional, concise (aim 120-180 words).
                - Use plain numbers with $ and 2 decimals (eg, $0.20, $400.00).

                Now execute PHASE A (tool calls) first, then produce PHASE B as the final answer.
                """
            )
            print("Sales Agent Reply: ", sales_reply)

            # Return customer reply back to orchestrator 
            return sales_reply

        super().__init__(
            tools=[handle_customer_request],
            model=model,
            name='Orchestrator',
            description="""You are an orchestrator that manages the sales of paper products.
            Call the tool handle_customer_request with the full customer request string. 
            Must include date of customer request in the argument string.
            """,
            # You coordinate between customer service, medical review, and claim processing agents.
        )
############# End of agent class declarations ##############

# Debug scenario to test individual components
def run_debug_scenario_1():
    """
    Function to debug various features of the workflow. 
    """
    print("Running Debug Scenario 1...")
    
    # Init DB with data 
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return
    print("Database initialized.")

    '''
    # Test stock level
    print("Item names: ", paper_supplies_item_names)
    item_names_to_search = ['Flyers', 'Poster paper', 'Invitation cards']
    stock_level_df = get_stock_level_multiple_items(item_names=item_names_to_search, as_of_date = datetime.now())
    print("Stock level: \n", str(stock_level_df))

    # Test search_quote_history
    search_terms = ["school", "colorful", "urgent"]
    search_results = search_quote_history_tfidf(search_terms=search_terms)
    print("Search terms: ", search_terms)
    print("Search results: ", search_results)
    
    # Test extract_items_from_request tool 
    test_request = "I need to order 10,000 sheets of A4 paper, 5,000 sheets of A3 paper, and 500 reams of printer paper. The supplies must be delivered by April 15, 2025, for our upcoming conference. Please confirm the order and delivery schedule."
    extracted_items = extract_items_from_request(test_request)
    print("Test request: ", test_request)
    print("Extracted items from request: ", extracted_items)
    '''

    # Initialize orchestrator 
    orchestrator = Orchestrator(model)
    sample_customer_request = """
    I am a restaurant manager organizing a large concert. I am making a request on (Date of request: 2025-04-17)
    I would like to place a large order for the 
        following paper supplies for our upcoming concert: 5,000 flyers, 2,000 posters, and 10,000 tickets. 
        We need these items delivered by May 15, 2025, to ensure adequate time for distribution and promotion. 
        Thank you! 
    """
    test_request_run = orchestrator.run(f"""
        Use the tool handle_customer_request to handle the customer request. Use the full customer request as string input. 
        Do not manipulate the below customer request when using the tool as parsing will be done in the tool itself. 
                                        
        Raw customer request: {sample_customer_request}
    """)
    print(test_request_run)

    # Get initial state
    '''
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    print("Report: ", report )
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    print(f"Initial Cash Balance: ${current_cash:.2f}")
    print(f"Initial Inventory Value: ${current_inventory:.2f}")
    '''

    # Print table contents for debugging 
    # print_table_contents("transactions", db_engine)
    # print_table_contents("inventory", db_engine)
    # print_table_contents("quote_requests", db_engine)
    # print_table_contents("quotes", db_engine)

# Run your test scenarios by writing them here. Make sure to keep track of them.
def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample = quote_requests_sample.dropna(how="all")
        quote_requests_sample["request_date"] = pd.to_datetime(
             quote_requests_sample["request_date"].astype(str).str.strip(),
            errors="coerce"  # invalid/blank -> NaT
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return
    print("Database initialized.")

    # Get initial state
    print(quote_requests_sample.head())
    print(quote_requests_sample.columns)
    quote_requests_sample = quote_requests_sample.dropna(how="all")
    min_dt = quote_requests_sample["request_date"].dropna().min()
    initial_date = min_dt.strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    results = []
    counter = 0 
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Format orchestrator request input string 
        request_with_date = f"{row['request']} (Date of request: {request_date})"
        orchestrator_input = f"I am a {row['job']} organizing a {row['need_size']} {row['event']} (event). I am making a request on {request_date}. {request_with_date} "
        print(f"Orchestrator input: {orchestrator_input}")
        # Eg: I am a restaurant manager organizing a large concert. I would like to place a large order for the 
        # following paper supplies for our upcoming concert: 5,000 flyers, 2,000 posters, and 10,000 tickets. 
        # We need these items delivered by May 15, 2025, to ensure adequate time for distribution and promotion. 
        # Thank you! (Date of request: 2025-04-17)

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############
        
        # Initialize orchestrator 
        
        orchestrator = Orchestrator(model)
        
        customer_reply = orchestrator.run(f"""
            Use the tool handle_customer_request to handle the customer request. Use the full customer request as string input. 
            Do not manipulate the below customer request when using the tool as parsing will be done in the tool itself. 
                                            
            Raw customer request: {orchestrator_input}""")
    
        # Update state
        response = customer_reply
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response, 
                "financial_report": report,
            }
        )

        # Debug break loop 
        counter += 1 
        if counter >= 1: 
          break
        
        time.sleep(1)
        
    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save final report 
    with open("final_report.json", "w") as f:
       json.dump(final_report, f)

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results

if __name__ == "__main__":
    # Debug scenario testing 
    # run_debug_scenario_1()

    # Run actual test cases 
    results = run_test_scenarios()

    # Save results 
    with open("results.json", "w") as f:
       json.dump(results, f)

