📌 Project Overview

This project focuses on analyzing credit card customer behavior and segmenting customers into meaningful groups using Machine Learning clustering techniques. The main objective is to identify customer spending patterns, payment behavior, credit usage habits, and potential financial risk groups.

By applying clustering algorithms such as K-Means and Agglomerative Clustering, customers were divided into distinct behavioral segments like high-value customers, cash-heavy users, low-engagement users, and moderate users.

The project also includes data visualization, feature engineering, PCA-based dimensionality reduction, and an interactive dashboard built using Streamlit.

🎯 Objectives
Analyze customer financial behavior using transaction data
Perform customer segmentation using clustering algorithms
Identify high-value and high-risk customer groups
Generate actionable business insights
Visualize customer segments through charts and dashboards
📊 Dataset Information
Dataset: Credit Card Customer Dataset
Records: ~8950 customers
Features: 18 behavioral and financial attributes
Important Features
BALANCE
PURCHASES
CASH_ADVANCE
CREDIT_LIMIT
PAYMENTS
PURCHASES_FREQUENCY
TENURE
MINIMUM_PAYMENTS
🛠️ Technologies & Tools Used
Programming Language
Python
Libraries
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Machine Learning Techniques
K-Means Clustering
Agglomerative Clustering
PCA (Principal Component Analysis)
Dashboard & Development
Streamlit
Jupyter Notebook
VS Code
🧠 Project Workflow
Data Collection
Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Clustering using K-Means
Cluster Evaluation
PCA Visualization
Dashboard Development
Business Insight Generation
📈 Visualizations Included

The project contains multiple visualizations to understand customer behavior and clustering performance.

Included Charts
Correlation Heatmap
Elbow Curve
Silhouette Score Analysis
Customer Cluster Scatter Plot
Feature Distribution Plots
Radar Chart
Boxplots
Cluster Comparison Charts
📂 Project Structure
Credit-Card-Customer-Segmentation/
│
├── images/
│   ├── boxplots.png
│   ├── boxplots_by_cluster.png
│   ├── cluster_barchart.png
│   ├── cluster_scatter.png
│   ├── correlation_heatmap.png
│   ├── distribution_plots.png
│   ├── elbow_curve.png
│   ├── radar_chart.png
│   └── silhouette_scores.png
│
├── CC GENERAL.csv
├── credit_card_clusters.csv
├── dashboard.py
├── index.html
├── Project.ipynb
├── CreditCard_Segmentation_Rachel.pptx
├── Phase01-Credit_Card_Segmentation_Report.docx
├── Phase02_Build_Report_Week2.docx
├── Phase03-Alva_Rachel_Wilfred_Internship_Report.docx
└── README.md
⚙️ Data Preprocessing

The following preprocessing steps were performed:

Handling missing values
Feature scaling and normalization
Removal of unnecessary columns
Data validation and cleaning
🔍 Feature Engineering

Additional behavioral metrics were created:

Credit Utilization Ratio
Payment Ratio
Spending Ratio

These features improved clustering performance and customer behavior analysis.

🤖 Machine Learning Models Used
🔹 K-Means Clustering

Used to segment customers into 4 behavioral groups.

Why K-Means?
Efficient for large datasets
Easy cluster interpretation
Suitable for customer segmentation
🔹 Agglomerative Clustering

Used for validating cluster structures and comparing segmentation performance.

📊 Cluster Segments Identified
🔹 High-Value Customers
High purchases and transaction frequency
Valuable for premium offers and loyalty programs
🔹 Cash-Heavy Users
Frequent cash advances
Higher financial risk indicators
🔹 Low Engagement Customers
Minimal purchases and low activity
Potential targets for customer engagement strategies
🔹 Moderate Users
Balanced spending and payment behavior
Stable customer segment
📉 Model Evaluation

The following evaluation techniques were used:

Elbow Method

Used to determine the optimal number of clusters.

Silhouette Score

Used to validate clustering quality and separation.

PCA Visualization

Used to reduce dimensionality and visualize clusters effectively.

🖥️ Dashboard Features

The project includes an interactive dashboard developed using Streamlit.

Dashboard Components
KPI Cards
Cluster Distribution
Spending vs Credit Analysis
Risk Insights
PCA Visualization
Interactive Filters
💼 Business Applications

This project can help financial institutions in:

Customer segmentation
Personalized marketing
Risk management
Customer retention
Financial behavior analysis
Fraud and anomaly detection
🚀 How to Run the Project
1️⃣ Clone Repository
git clone <your-repository-link>
2️⃣ Install Required Libraries
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
3️⃣ Run Jupyter Notebook

Open:

Project.ipynb

Run all cells sequentially.

4️⃣ Run Dashboard
streamlit run dashboard.py
📌 Key Insights
Customer behavior varies significantly across segments
High-value users contribute most to revenue
Cash-heavy users may indicate higher financial risk
Data-driven segmentation improves business decision-making
🔐 Future Enhancements
Real-time customer analytics
Fraud detection integration
Advanced clustering techniques
Power BI dashboard integration
Cloud deployment
🏁 Conclusion

This project demonstrates how machine learning and data analytics can be applied in the financial domain to identify meaningful customer segments and generate actionable business insights.

The segmentation model and dashboard together help improve marketing strategies, customer engagement, and financial risk management using data-driven approaches.

👩‍💻 Author
Rachel Alva

AI & Data Analytics Intern
Rooman Technologies
