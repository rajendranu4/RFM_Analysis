import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class RFMAnalysis:
    def __init__(self, df, unique_id, date_column_name, quantity_column_name, price_column_name):
        self.df = df
        self.uid = unique_id
        self.r_name = date_column_name
        self.f_name = quantity_column_name
        self.m_name = price_column_name
        self.current_date = pd.Timestamp(year=2012, month=1, day=1)  # sample assumed date when the analysis was done
        self.rfm_values = pd.DataFrame()

    def recency_calc(self):

        ''' Recency is calculated by taking the difference between current date and the recent (max) date
            out of all purchases that the customer made. The sample date assumed current date is 1-1-2012'''
        self.r_values = (self.current_date - self.df.groupby(self.uid).agg({self.r_name: max}))[self.r_name].apply(
            lambda date: date.days)
        # print("\nRecency Calculation")
        # print(self.r_values)

    def frequency_calc(self):

        ''' Frequency is calculated by counting the number of times the customer purchased product
            i.e number of invoices created for a particular customer becomes the frequency for that customer'''
        self.f_values = self.df.groupby(self.uid).agg({"InvoiceNo": "nunique"}).reset_index()
        # print("\nFrequency Calculation")
        # print(self.f_values)

    def monetory_calc(self):

        ''' Monetary is calculated by summing the price spent by the customer'''
        self.df['TotalPrice'] = self.df[self.f_name] * self.df[self.m_name]
        self.m_values = self.df.groupby(self.uid).agg({"TotalPrice": "sum"}).reset_index()
        # print("\nMonetory Calculation")
        # print(self.m_values)

    def rfm_values_calc(self):
        print("\nPrinting RFM Values")

        ''' Recency, Frequency, Monetary dataframes are merged by CustomerID to for a single RFM values table'''
        self.rfm_values = pd.merge(self.r_values, self.f_values, on=['CustomerID'])
        self.rfm_values = pd.merge(self.rfm_values, self.m_values, on=['CustomerID'])
        self.rfm_values = self.rfm_values.rename(
            columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetory"})
        print(self.rfm_values)

    def rfm_scores_calc(self):

        ''' Recency, Frequency, Monetary scores are calculated using quintile method where all the values
            are separated into 5 quintiles and each 5 quintile labeled 1 to 5'''

        quintiles_total = 5
        labels_asc = [1, 2, 3, 4, 5]
        labels_desc = [5, 4, 3, 2, 1]

        # Recency is labelled from 1 to 5, 5 being the very recent
        self.rfm_values['R'] = pd.qcut(self.rfm_values["Recency"], quintiles_total, labels=labels_desc)

        # Frequency is labelled from 1 to 5, 1 being low frequency and 5 being high frequency
        # bunch of equal frequency values are ranked in order they appear in the column
        self.rfm_values['F'] = pd.qcut(self.rfm_values["Frequency"].rank(method="first"), quintiles_total,
                                       labels=labels_asc)

        # Monetary is labelled from 1 to 5, 1 being low and 5 being high
        self.rfm_values['M'] = pd.qcut(self.rfm_values["Monetory"], quintiles_total, labels=labels_asc)

        # self.rfm_scores['F+M'] = (self.rfm_scores['F'] + self.rfm_scores['M']) / 2

        self.rfm_values['rfm'] = self.rfm_values['R'].astype(str) + \
                                 self.rfm_values['F'].astype(str) + \
                                 self.rfm_values['M'].astype(str)
        print("\nPrinting RFM Scores")
        print(self.rfm_values)

    def segment_customers(self):

        ''' Customers are segmented to one of 11 segments. Ex. Customer with values of Recency - 4 or 5,
            Frequency - 4 or 5 and Monetary - 3 to 5 are assigned to Champions as they have high RFM values'''
        for i, row in self.rfm_values.iterrows():
            if re.match(r'[4-5][4-5][3-5]', row['rfm']):
                self.rfm_values.at[i, 'Segment'] = 'Champions'

            elif re.match(r'[3-5][3-4][4-5]', row['rfm']):
                self.rfm_values.at[i, 'Segment'] = 'Loyal Customers'

            elif re.match(r'[3-5][1-3][3-5]', row['rfm']):
                self.rfm_values.at[i, 'Segment'] = 'Potential Loyalist'

            elif re.match(r'[4-5][1-2][1-5]', row['rfm']):
                self.rfm_values.at[i, 'Segment'] = 'Recent Customers'

            elif re.match(r'[3-5][1-5][1-2]', row['rfm']):
                self.rfm_values.at[i, 'Segment'] = 'Promising'

            elif re.match(r'[3-4][3-4][3-4]', row['rfm']):
                self.rfm_values.at[i, 'Segment'] = 'Customers Needing Attention'

            elif re.match(r'[2-3][1-3][1-3]', row['rfm']):
                self.rfm_values.at[i, 'Segment'] = 'About to Sleep'

            elif re.match(r'[1-3][3-5][1-5]', row['rfm']):
                self.rfm_values.at[i, 'Segment'] = 'At Risk'

            elif re.match(r'[1-2][4-5][4-5]', row['rfm']):
                self.rfm_values.at[i, 'Segment'] = 'Cant Lose Them'

            elif re.match(r'[1-2][1-2][3-5]', row['rfm']):
                self.rfm_values.at[i, 'Segment'] = 'Hibernating'

            elif re.match(r'[1-2][1-2][1-2]', row['rfm']):
                self.rfm_values.at[i, 'Segment'] = 'Lost'

            else:
                self.rfm_values.at[i, 'Segment'] = 'Unidentified'

        # print("\nPrinting Customer Segments")
        # print(self.rfm_scores[self.rfm_scores['Segment']=='Unidentified'])
        # print(self.rfm_values)

        return self.rfm_values

    def business_intelligence(self):

        ''' Percentage of customers in each segment is visualized using pie chart'''
        chart_sections = self.rfm_values['Segment'].value_counts()
        chart_sections_dict = chart_sections.to_dict()
        # del chart_sections_dict['Unidentified']
        print("\nCount")
        print(chart_sections)
        # print(chart_sections_dict)

        explode_param = np.zeros(len(chart_sections_dict))
        explode_param[0] = 0.1
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.pie([float(chart_sections_dict[value]) for value in chart_sections_dict],
                labels=[label for label in chart_sections_dict],
                autopct='%1.0f%%',
                pctdistance=0.7,
                labeldistance=1.1,
                shadow=True,
                explode=explode_param,
                )
        plt.show()

        ''' Questions on customer segmentation for business intelligence can be answered with the following data
            For ex. who are the best customers, who are the verge of churning, who are loyal customers etc. '''
        # best customers
        best_customers = self.rfm_values[self.rfm_values['Segment'] == 'Champions']

        print("\n1) Below are the BEST CUSTOMERS")
        print(best_customers)

        # customers at the verge of churning
        churn_customers = self.rfm_values[self.rfm_values['Segment'] == 'About to Sleep']
        print("\n2) Below are the CUSTOMERS WHO ARE AT THE VERGE OF CHURNING")
        print(churn_customers)

        # potential profitable customers
        profitbale_customers = self.rfm_values[self.rfm_values['Segment'] == 'Potential Loyalist']
        print("\n3) Below are the CUSTOMERS WHO CAN BE CONVERTED TO PROFITABLE CUSTOMERS")
        print(profitbale_customers)

        # lost customers
        lost_customers = self.rfm_values[self.rfm_values['Segment'] == 'Lost']
        print("\n4) Below are the CUSTOMERS WHO ARE LOST AND NO NEED TO PAY ATTENTION ON THEM")
        print(lost_customers)

        # customers need to be retained
        retain_customers = self.rfm_values[self.rfm_values['Segment'] == 'At Risk']
        print("\n5) Below are the CUSTOMERS WHO ARE NEEDED TO BE RETAINED")
        print(retain_customers)

        # Loyal Customers and customers who are likely to respond to campaigns
        loyal_customers = self.rfm_values[self.rfm_values['Segment'] == 'Loyal Customers']
        print("\n6)and 7) Below are the LOYAL CUSTOMERS AND CUSTOMERS WHO ARE LIKELY TO RESPOND TO CAMPAIGNS")
        print(loyal_customers)

    def clusters_customer_segments(self):

        ''' Visualizing the customers segments with RFM using K-Means algorithm'''
        clusters_data = self.rfm_values[["Recency", "Frequency", "Monetory"]]
        print(clusters_data)

        scaler = StandardScaler()
        Scaled_RFM_data = scaler.fit_transform(clusters_data)

        kmeans = KMeans(n_clusters=5, max_iter=200, random_state=35)

        kmeans.fit(Scaled_RFM_data)
        y_kmeans = kmeans.predict(Scaled_RFM_data)

        plt.scatter(Scaled_RFM_data[:, 0], Scaled_RFM_data[:, 1], c=y_kmeans, s=50, cmap='viridis')