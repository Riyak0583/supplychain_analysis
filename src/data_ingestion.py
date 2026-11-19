
import pandas as pd

# Step 1: Data Ingestion
def data_ingestion():

    '''
    Dataset: Supply Chain and Logistics

    Target Column (Y):
    Product_Demand

    Independent Columns (X):

    Customer Features:
    Gender, Age, Marital_Status, City, State,
    Customer_Segment, Membership_Type, Previous_Orders,
    Purchase_Frequency, Is_Repeat_Customer

    Product Features:
    Category, Sub_Category, Brand,
    Product_Name, Product_Price, Product_Cost,
    Discount_Applied, Promotion_Type

    Sales Features:
    Sales_Channel, Payment_Method, Order_Quantity,
    Total_Sales, Profit_Margin, Revenue

    Transaction Features:
    Transaction_ID, Order_Date, Order_Status

    Delivery Features:
    Shipping_Method, Delivery_Status, Delivery_Time,
    Shipping_Cost, Warehouse_Location

    Inventory Features:
    Stock_Level, Reorder_Level, Supplier_ID,
    Supplier_Lead_Time, Inventory_Turnover

    Logistics Features:
    Distance, Transport_Mode, Vehicle_Type,
    Fuel_Cost, Route_Information
    '''

    df = pd.read_csv(r"C:\SupplyChain_Analysis\data\raw\SupplyChain_Dataset.csv")

    return df


# Calling the function
df = data_ingestion()

# Checking the dataset
print(df.head())
print(df.shape)
print(df.info())
print(df.drop_duplicates())