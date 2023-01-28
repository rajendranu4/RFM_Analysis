import pandas as pd
from RFM import RFMAnalysis

if __name__ == '__main__':
    df = pd.read_excel('Online_Retail.xlsx')

    print("\nTotal number of rows and number of features in the dataset")
    print(df.shape)

    print("\nChecking if the dataset contains null values")
    print(df.isnull().sum())

    ''' Removing rows that has no CustomerID values since RFM is based on the customers
        and since CustomerID is unique to every customer, imputation cannot be done for a row that is missing CustomerID '''
    df = df[df['CustomerID'].notna()]

    print("\nChecking if the dataset contains null values after removing rows that has missing CustomerIDs")
    print(df.isnull().sum())

    print("\nTotal number of rows and number of features in the dataset currently")
    print(df.shape)

    ''' cancelled invoices are separately stored in another dataframe
        cancelled invoices are identified by InvoiceNo starting with C '''
    df_inv_c = df[df['InvoiceNo'].apply(lambda x: str(x).startswith('C'))]

    # cancelled invoices are removed from the current dataframe
    df = df[df['InvoiceNo'].apply(lambda x: not str(x).startswith('C'))]

    print("\nTotal number of rows and number of features in the dataset currently")
    print(df.shape)

    rfm_analyser = RFMAnalysis(df, 'CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice')
    rfm_analyser.recency_calc()
    rfm_analyser.frequency_calc()
    rfm_analyser.monetory_calc()
    rfm_analyser.rfm_values_calc()
    rfm_analyser.rfm_scores_calc()
    rfm = rfm_analyser.segment_customers()
    rfm_analyser.business_intelligence()
    rfm_analyser.clusters_customer_segments()