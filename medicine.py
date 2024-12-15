import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import silhouette_score

# Title 
st.title('K-Means Clustering for Medicine')
st.image("C:/Users/User/Downloads/resep-obat.jpg", use_column_width=True)

# File Upload
uploaded_file = st.file_uploader("Upload a CSV to perform K-Means Clustering on medicine data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Preview data
    st.write('Data Preview: ')
    st.write(data.head())

    # Data summary
    st.write('Dataset Summary: ')
    st.write(data.describe())

    # Handling missing values
    st.subheader('Handling Missing Values')
    st.write('Original data shape:', data.shape)
    st.write('Number of Missing values before handling', data.isnull().sum())

    # Drop rows with any NaN values
    data.dropna(inplace=True)

    # Verify data integrity post handling missing values
    st.write('Number of missing values after handling', data.isnull().sum().sum())

    # Data Preprocessing
    st.subheader('Data Preprocessing')
    columns = data.columns.tolist()

    # Convert string columns to integers (for non-numeric data)
    for col in columns:
        if col != 'medicine_name' and data[col].dtype == 'object':
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])

    selected_columns = st.multiselect('Select columns for clustering', columns)

    if selected_columns:
        st.write(f'Selected columns for clustering: {selected_columns}')
        
        # Visualize data distribution for selected columns
        st.subheader('Data Distribution')
        for col in selected_columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data[col], kde=True, ax=ax, bins=50)
            ax.set_title(f'Distribution of {col}')
            st.pyplot(fig)

        if len(selected_columns) >= 2:
            # Normalize the selected columns
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data[selected_columns])

            # Check and remove outliers using Z-score method
            st.subheader('Checking and Removing Outliers')

            # Calculate Z-scores
            z_scores = np.abs(zscore(scaled_data))
            outliers = (z_scores > 3).all(axis=1)

            st.write(f"Number of outliers detected: {sum(outliers)}")
            
            # Remove outliers from the data
            data_cleaned = data[~outliers]
            st.write(f"Shape of data after removing outliers: {data_cleaned.shape}")

            # Visualize outliers with Boxplot
            st.subheader('Boxplot for Outlier Detection')

            # Create a boxplot for each selected column
            fig, axes = plt.subplots(1, len(selected_columns), figsize=(15, 5))
            if len(selected_columns) == 1:
                axes = [axes] 

            for i, col in enumerate(selected_columns):
                sns.boxplot(data=data, x=col, ax=axes[i])
                axes[i].set_title(f'Boxplot of {col}')

            st.pyplot(fig)

            # PCA (Principal Component Analysis) for dimensionality reduction
            st.subheader('PCA (Principal Component Analysis)')

            # Apply PCA if more than one feature is selected
            if len(selected_columns) > 1:
                pca = PCA(n_components=2)  
                pca_data = pca.fit_transform(data_cleaned[selected_columns])

                # Visualize the PCA result
                pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
                st.write("PCA Results (2 Components):")
                st.write(pca_df.head())

                # Plot the PCA result
                fig, ax = plt.subplots()
                sns.scatterplot(x='PC1', y='PC2', data=pca_df, ax=ax)
                st.pyplot(fig)

            # Clustering (K-Means)
            st.subheader('Clustering')
            num_clusters = st.slider('Select number of clusters', 2, 10, 3)
            kmeans = KMeans(n_clusters=num_clusters)

            # Perform KMeans clustering on the cleaned data (after removing outliers)
            scaled_cleaned_data = scaler.fit_transform(data_cleaned[selected_columns])  
            kmeans_predict = kmeans.fit_predict(scaled_cleaned_data) 
            data_cleaned['Cluster'] = kmeans_predict  

            # Display the first few rows of 'Cluster' along with 'medicine_name' and the selected columns
            if 'medicine_name' in data_cleaned.columns:
                st.write('Cluster with Medicine Names and Selected Columns:')
                st.write(data_cleaned[['medicine_name'] + selected_columns + ['Cluster']].head())  # Show 'MedicineName', 'Cluster', and selected columns
            else:
                st.write('Medicine name column not found. Displaying only the cluster and selected columns:')
                st.write(data_cleaned[['Cluster'] + selected_columns].head())  # Show only the cluster and selected columns

            # Visualize Clusters 
            st.subheader('Cluster Visualization')
            if len(selected_columns) >= 2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=data_cleaned[selected_columns[0]], y=data_cleaned[selected_columns[1]], hue=data_cleaned['Cluster'], palette='viridis', ax=ax)
                st.pyplot(fig)


            # Evaluate Clustering
            st.subheader('Clustering Evaluation')

            # Silhouette Score 
            silhouette_avg = silhouette_score(scaled_cleaned_data, kmeans_predict)
            st.write(f'Silhouette Score: {silhouette_avg:.3f}')
            
            # Inertia
            inertia = kmeans.inertia_
            st.write(f'Inertia: {inertia:.3f}')

        else:
            st.write('Please select at least two columns for clustering visualization')

    else:
        st.write('Please select columns for clustering')

    # Footer
    st.markdown('---')
    st.title('Thank You')
