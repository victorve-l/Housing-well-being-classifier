# Project Overview


In Mexico, the National Household Income and Expenditure Survey (ENIGH) is carried out every two years to provide detailed information about income distribution, living conditions, and household expenditure. While the survey offers rich and valuable data, classifying households into socioeconomic strata based on multiple variables such as housing conditions, access to services, and education remains a complex task.

This project leverages machine learning techniques to simplify this process by creating a Housing Well-being Classifier, which uses household and population characteristics to predict socioeconomic levels in a transparent and data-driven way.

# Project Motivation

The Housing Well-being Classifier aims to predict the socioeconomic stratum of households based on housing conditions, demographic factors, and access to essential services. Understanding housing well-being is important because it reflects inequalities in living conditions, access to education, health, and infrastructure. By developing a predictive model, this project seeks to transform survey data into actionable insights, which can help policymakers, researchers, and organizations identify vulnerable groups and design more effective social programs.

This project is not only a technical exercise in data science but also an applied example of how machine learning can support evidence-based decision making in social and economic contexts.


<div align="center"><img src="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/Templates/Logo.png" width="400" height="400" /></div>



<h2 id="project-structure">Project Structure</h2>

1.- <a href="#introduction">Introduction ✨</a>



2.- <a href="#python-libraries">Tooling & Libraries 🛠️</a>



3.- <a href="#data-collection-wrangling">Data Importation and Transformation 🧹</a>



4.- <a href="#exploratory-data-analysis">Exploratory Data Analysis (EDA) 🧐</a>



5.- <a href="#data-visualization">Decision Trees Modeling 🌳</a>



6.- <a href="#segmentation-k-means">Random Forest Implementation 🚀</a>



7.- <a href="#time-series-forecasting">Findings & Conclusions 💡</a>






<br>

<h2 id="introduction">Introduction ✨</h2>

<p>This project develops a Housing Well-being Classifier using data from the Mexican ENIGH survey. The goal is to predict household socioeconomic levels based on housing and living conditions, providing a practical application of machine learning in a real-world social context.</p>

<div style="display: flex; align-items: center;">

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/Templates/portada_eneigh.jpg" width="500" height="500" />

<p><a href="#project-structure">Back to Project Structure</a></p>
</div>



<h2 id="python-libraries">Tooling & Libraries 🛠️</h2>

• Pandas  <a target="_blank" href="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" style="display: inline-block;"><img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="20" height="20" /></a>



• NumPy  <a target="_blank" href="https://numpy.org/doc/stable/_static/numpylogo.svg" style="display: inline-block;"><img src="https://numpy.org/doc/stable/_static/numpylogo.svg" alt="numpy" width="30" height="30" /></a>



• Scikit-learn  <a target="_blank" href="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" style="display: inline-block;"><img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="30" height="30" /></a>



• Statsmodels  <a target="_blank" href="https://www.statsmodels.org/stable/_static/statsmodels_logo.svg" style="display: inline-block;"><img src="https://www.statsmodels.org/v0.11.1/_images/statsmodels-logo-v2-no-text.svg" alt="statsmodels" width="20" height="20" /></a>



• Matplotlib  <a target="_blank" href="https://matplotlib.org/_static/logo2_compressed.svg" style="display: inline-block;"><img src="https://matplotlib.org/_static/logo2_compressed.svg" alt="matplotlib" width="28" height="28" /></a>



• Seaborn  <a target="_blank" href="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" style="display: inline-block;"><img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="24" height="24" /></a>




<p><a href="#project-structure">Back to Project Structure</a></p>

<h2 id="data-collection-wrangling">Data Importation and Transformation 🧹</h2>

<p>The data was obtained from the information programs section of INEGI and also from the microdata section of ENEIGH (https://www.inegi.org.mx/programas/enigh/nc/2024/). This section contains several tables in .csv format describing the socioeconomic characteristics of households, and the following were selected for this analysis:</p>
<ul style="font-size: 0.9em;">
  <li><strong>• HOGARES :</strong> Characteristics of the households in which the members of the household live.
  <li><strong>• INGRESOS:</strong> Income and financial and capital gains of household members</strong></li>
  <li><strong>• VIVIENDAS:</strong> Characteristics of the dwellings occupied by household members</strong></li>
  <li><strong>• POBLACION:</strong> Sociodemographic characteristics of household members</strong></li>
  <li><strong>• GASTOS_HOGAR:</strong> Household expenditure</strong></li>
    </strong>
</ul>
    
<p>This section on data import and transformation was divided into part 1 and part 2. In the first part, the most important variables from each table were selected. This part had a huge relevance, as some variables provide more information than others. For example, per capita income gives us much more information than the musical preferences of household members. 

After variable selection, several pandas methods were used to correct typos, convert data types (string to int), remove duplicate values, remove NaN values, among others. The most commonly used methods for making these modifications were as follows:</p>
<ul style="font-size: 0.9em;">
  <li><strong>• rename:</strong> rename columns or index labels. </li>
  <li><strong>• replace:</strong> replace specified values. </li>
  <li><strong>• astype:</strong> cast a pandas object to a specified dtype. </li>
  <li><strong>• groupby:</strong> group DataFrame using a mapper or by a series of columns. </li>
  <li><strong>• round:</strong> round rows to a specified number of decimals. </li>
  <li><strong>• isin:</strong> verify whether the elements of a dataFrame are present in a set of values (list, dictionary, etc.) and return a boolean result. </li>
  <li><strong>• fillna:</strong> fill missing values with a specified value. </li>
  <li><strong>• drop:</strong> eliminate a specific column or specific rows. </li>
  <li><strong>• isna:</strong> identify missing values. </li>
  <li><strong>• duplicate:</strong> identify duplicate values.</li>
</ul>
    
<a target="_blank" href="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/01_Data%20Importation%20and%20Transformation%20Part%201/01.%20Data_Importation_and_Transformation_Part1.ipynb"> Full code for this section (Part 1)</a>


<p>Once the variables were selected, transformed, and consolidated into a single dataframe, some calculations and conversions were performed in the second part to facilitate subsequent analyses. Below is a summary of the changes made in this part:</p>
<ul style="font-size: 0.9em;">
  <li><strong>• Calculation of average income per household:</strong> the average was calculated based on monthly data and the amount recorded for each inhabitant. The data was saved in the variable average_income.</li>
  <li><strong>• Conversion of binary variables:</strong> </li>
  <li><strong>• Sum of electronic devices: </strong> a variable (total_devices) was created to add up the total number of electronic devices present in the home. Originally, the dataframe contained variables that gave us information such as the number of cell phones, number of laptops, number of computers, etc. per household.</li>
  <li><strong>• Application of One-Hot Encoding:</strong> one-hot encoding is a technique used to convert categorical variables into a numerical format that machine learning algorithms can understand. Several of the categorical variables present were converted to numerical variables.</li>
  <li><strong>• Application of the SMOTE technique:</strong> since we had an unbalanced number of records, we applied SMOTE to balance the number of records between classes. Instead of duplicating records from minority classes, SMOTE creates new, synthetic data based on existing records.</li>
  <li><strong>• Renaming values in the socioeconomic stratum variable (est_socio):</strong> originally, this variable was coded with numbers from 1 to 4. In this dataframe, the data was renamed with strings to represent the socioeconomic stratum as “low socioe,” “lower-middle,” “upper-middle,” and “high,” respectively.</li>
  <li><strong>• Renaming of entities:</strong> mexican federal entities (states) were coded with numbers and were replaced with their respective names. </li>
</ul>

<p>At the end of this part, a consolidated dataframe was obtained with 32 variables transformed and ready for further analysis:</p>
<img src="https://raw.githubusercontent.com/victorve-l/Housing-well-being-classifier/main/Templates/gif_dataframe.gif" alt="GIF de un DataFrame" width="500"/>

<a target="_blank" href="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/02_Data%20Importation%20and%20Transformation%20Part%202/02.%20Data_Importation_and_Transformation_Part2.ipynb"> Full code for this section (Part 2)</a>
<p><a href="#project-structure">Back to Project Structure</a></p>




<h2 id="exploratory-data-analysis">Exploratory Data Analysis (EDA) 🧐</h2>

<p>In the exploratory data analysis (EDA), our first goal was to see if our dataset is balanced or if there are some socioeconomic classes with few cases.
<div align="center"><img src="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/Templates/01_socioeconomic_stratum_distribution.png" width="600" height="500" /></div>

According to the graph, our data showed an equal number of households according to socioeconomic stratum. This is the magic of SMOTE, the technique we previously applied to balance our number of records according to the socioecnomic stratum. </p>

<p>Our 2nd question was how different is the average income between stratums? We constructed a boxplot using seaborn:

<div align="center"><img src="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/Templates/02_average_income_by_socioeconomic-stratum.png" width="600" height="500" /></div>
Although it may seem obvious that the upper stratum tends to have higher incomes than the lower stratum, it is necessary to confirm this data. Now the question is: Why are there people in the lower socioeconomic stratum who have high average incomes and are still considered lower class? It should be noted that the socioeconomic stratum defined by the 2010 Population and Housing Census takes into account other variables related to housing characteristics to define the type of class. This is why there are people in the lower stratum who may have incomes similar to those of someone in the upper stratum.</p>

<p>Next, we made a stacked bar chart to see how floor material changes depending on the socioeconomic stratum:
<div align="center"><img src="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/Templates/03_floor_material_by_socioeconomic_stratum.png" width="600" height="500" /></div>

According to the stacked bar chart, the household floor material differs between socioeconomic strata. Dirt material is practically only present in low and lower-middle stratums, while wood or mosaic material are more common in higher stratums. </p>

p>Finally, we wanted to compare car pocession according to socioenomic stratum: 
<div align="center"><img src="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/Templates/04_car_pocession_according_to_socioeconomic_stratum.png" width="600" height="500" /></div>
According to the graph, the high socioeconomic stratum has around 3 to 4 cars, while the low stratum has practically 0 cars per household.</p>


<a target="_blank" href="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/03_Exploratory%20Data%20Analysis/03.%20EDA.ipynb"> Full code for this section</a>
<p><a href="#project-structure">Back to Project Structure</a></p>




<h2 id="data-visualization">Decision Trees Modeling 🌳</h2>

<p>Data visualization related to reckit sales was performed using the matplotlib and seaborn libraries. According to the analysis, Vanish products (pre-washers and bleaches) have high and variable unit sales (products have been sold a higher number of times). In the case of Lysol, unit sales proved to be low but consistent (products have been sold a few times). </p>

<div align="center"><img src="https://github.com/victorve-l/Reckitt_EBAC/blob/main/Templates/Fig3.jpg?raw=true" width="350" height="400" /></div>

<p> Sales by region showed a distribution skewed to the right, meaning that in all regions products are sold in small quantities. Region 0 had the greatest variety of unit sales, while regions 2, 3, and 6 had the lowest concentration of sales. </p>

<div align="center"><img src="https://github.com/victorve-l/Reckitt_EBAC/blob/main/Templates/Fig6.jpg" width="600" height="400" /></div>

<p>With respect to the earnings performance of Reckitt products, the Vanish brand showed an increase in earnings between 2022-23, while earnings for Lysol products remained stable over the same period. </p>

<div align="center"><img src="https://github.com/victorve-l/Reckitt_EBAC/blob/main/Templates/Fig4a.jpg" width="700" height="280" /></div>

<p> Finally, an interactive dashboard was created in PowerBI where the total sales and profits of the Vanish and Lysol products, respectively, can be visualized. It is also possible to visualize total profits by region and by product type.</p>
<div align="center"><img src="https://github.com/victorve-l/Reckitt_EBAC/blob/main/Templates/Fig5.jpg" width="700" height="220" /></div>

<p><a target="_blank" href="https://github.com/victorve-l/Reckitt_EBAC/blob/main/03_Data%20Visualization/Data%20Visualization.ipynb"> Full code for this section</a></p>
<p><a target="_blank" href="https://github.com/victorve-l/Reckitt_EBAC/blob/main/03_Data%20Visualization/Power_BI_Dashboard.pdf"> Dashboard Visualization</a></p>
<p><a href="#project-structure">Back to Project Structure</a></p>




<h2 id="segmentation-k-means">Random Forest Implementation 🚀</h2>

<p>The K-means clustering algorithm was used to segment the products based on key variables such as total sales, total profits, product type, etc. Data was transformed using the following tools from scikit learn:</p>
<ul style="font-size: 0.9em;">
  <li><strong>• Target Encoder:</strong> transformation of categorical variables into numerical variables</li>
  <li><strong>• Standard Scaler :</strong> data standarization</strong></li>
  <li><strong>• PCA :</strong> Dimension reduction</strong></li>
</ul>

<p>The elbow method was applied to determine the optimal number of clusters. According to the graph, the optimal number is k=5.</p>
<div align="center"><img src="https://github.com/victorve-l/Reckitt_EBAC/blob/main/Templates/Fig7.jpg" width="500" height="400" /></div>

<p>Five clusters were identified using data reduced to 3 dimension by PCA and a value ok k=5. The clusters appear to be related to the number of units sold and total profits.</p>
<div align="center"><img src="https://github.com/victorve-l/Reckitt_EBAC/blob/main/Templates/Fig8.jpg" width="600" height="500" /></div>
<p>According to the cluster assignment, clusters 1 and 4 belong to products whose profits are 'low' (less than 1500 units). Cluster 3 belongs to products with intermediate profits (approximately 27 000 units). Clusters 0 and 2 belong to products that generated high profits (greater than 70,000 units). It should be noted that sales en these last groups were variable compared to others.</p>
<div align="center"><img src="https://github.com/victorve-l/Reckitt_EBAC/blob/main/Templates/Fig9.jpg" width="600" height="450" /></div>
<p><a href="#project-structure">Back to Project Structure</a></p>
<p><a target="_blank" href="https://github.com/victorve-l/Reckitt_EBAC/blob/main/04_Segmentation%20with%20K-means%20Clustering/Segmentation%20with%20K-means%20Clustering.ipynb">Full code for this section</a></p>







<h2 id="time-series-forecasting">Findings & Conclusions </h2>

<p>An ARIMA time series model was used to predict earnings, based on sales patterns observed in historical data. The variable TOTAL_VALUE_SALES was used, which represents the total value generated. A graph of historical Reckitt product earnings is shown below: </p>
<div align="center"><img src="https://github.com/victorve-l/Reckitt_EBAC/blob/main/Templates/Fig10.jpg" width="500" height="400" /></div>

<p>Before performing the forecasts with ARIMA, the calculation of differencing between observations (Δyt=yt-yt-1) was applied and the Dickey-Fuller test was applied to obtain the optimal p value (optimal p=3):</p>
<div align="center"><img src="https://github.com/victorve-l/Reckitt_EBAC/blob/main/Templates/Fig11.jpg" width="500" height="400" /></div>

<p>To determine the hyper parameters of the ARIMA model, the Akaike Information Criterion (AIC) was used to compare different statistical models and select the most appropriate model (more detail on code). The ARIMA (3,1,1) model had the best score according to the AIC, and this model was used to make point predictions using the test database (orange line in the graph) and confidence intervals of the predictions using the test database.</p>
<div align="center"><img src="https://github.com/victorve-l/Reckitt_EBAC/blob/main/Templates/Fig12.jpg" width="500" height="400" /></div>

<p>A calculation of the point forecasts and confidence intervals over the next 8 weeks was also made using the ARIMA (3,1,1) model (more detail on code). Finally, to evaluate the accuracy of the model, the mean square error (MSE) and the mean absolute percentage error (MAPE) were calculated:</p>
<ul style="font-size: 0.9em;">
  <li><strong>• RMSE:</strong> 3724.72</li>
  <li><strong>• MAPE:</strong> 7.98%</strong></li>
</ul>

<p>According to the ARIMA (3,1,1), a model with an RMSE of 3748 units and a MAPE of 8% was obtained, suggesting that this model is adequate to predict the future earnings of VANISH and LYSOL products.</p>
<p><a href="#project-structure">Back to Project Structure</a></p>
<p><a target="_blank" href="https://github.com/victorve-l/Reckitt_EBAC/blob/main/05_Time%20Series%20Forecasting/Time%20Series%20Forecasting.ipynb">Full code for this section</a></p>

<p><a target="_blank" href="https://github.com/victorve-l/Reckitt_EBAC/blob/main/Templates/Reckitt_DataScience_FinalPresentation.pdf">Link to presentation</a></p>
<p><a href="#project-structure">Back to Project Structure</a></p>
