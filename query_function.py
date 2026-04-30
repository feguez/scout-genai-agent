import sqlite3

def query_transactions(sql_query: str):
    """
    Use this tool when the user asks questions that require querying transaction data.
    The tool executes a SQLite query on the company database and returns the results.
    The database has one table called raw_transactions with these columns:
    - InvoiceNo (TEXT)
    - StockCode (TEXT)
    - Description (TEXT)
    - Quantity (INTEGER)
    - InvoiceDate (TIMESTAMP)
    - UnitPrice (REAL)
    - CustomerID (REAL)
    - Country (TEXT)
    - LineRevenue (REAL)
    Only SELECT queries are allowed.
    """
    print("The agent is using query_transactions tool")
    print(sql_query)

    # do not allow the agent to execute queries that don't start with SELECT
    if not sql_query.strip().upper().startswith("SELECT"):
        return "Only SELECT queries are allowed."

    # connect to the database
    conn = sqlite3.connect("tool_files/company.db")
    try:
        # execute the query and save the result as a dataframe
        df = pd.read_sql_query(sql_query, conn)
        return df.to_string(index=False)
    # if executing the query returns error this Exception code will
    # tell the LLM that the query was wrong but the Agent will not break
    except Exception as e:
        return f"Query error: {str(e)}"
    finally:
        conn.close() # close the connection to the database