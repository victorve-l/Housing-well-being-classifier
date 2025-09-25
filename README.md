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

<p>In this part, we begin our predictive analysis by applying Decision Trees as the first classification algorithm. Decision Trees are an excellent starting point because they are intuitive, easy to interpret, and capable of handling both numerical and categorical variables without extensive preprocessing. Moreover, they allow us to visualize how different household and housing characteristics contribute to socioeconomic stratification, which aligns with the objective of building a classifier for household living conditions. Although more complex models were used in further sections, Decision Trees provide a solid baseline and valuable insights into the structure of our data. </p>

<p>To build our decision tree model, we explored two node-splitting criteria:

<li><strong>Gini:</strong> measures the impurity of a node; lower values indicate that the samples mostly belong to a single class.</li>
<li><strong>Entropy:</strong> measures the uncertainty of a node using information theory; it seeks divisions that maximize the reduction in uncertainty.</li>
</p>


<p>Additionally, we used the max_depth parameter with different values. This allowed us to control the maximum depth of the tree and avoid overfitting, ensuring the model generalized better to new data. By testing different values, we found a balance between complexity and performance. Here, we present the metrics for the best results using gini and entropy criteria:

<div align="center"><img src="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/04_Decision%20Trees/gini_entropy_criteria.jpg" width="800" height="700" /></div>
</p>


<p>According to metrics such as precision, recall, F1-score, and accuracy, the ideal maximum depth is equal to 3 for both criteria. For practical purposes, the Gini criterion was used to create the decision tree. Below is the decision tree created with the graphviz module: 
<div align="center"><img src="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/04_Decision%20Trees/stratum_tree.png" width="800" height="700" /></div
</p>

<p>Let's dive a little deeper into class classification according to the decision tree using the Gini criterion:
  
<li><strong>Low stratum stratum:</strong> For a household to be classified as Low stratum, two key conditions must be met: The household has 3.5 or fewer electronic devices in total (total_dispositivos <= 3.5) and it has 0.5 or fewer cars (num_auto <= 0.5). If these two conditions are met, the model predicts with high certainty that the household is in the Low stratum.</li>
<li><strong>Lower-middle stratum:</strong> The classification for this stratum is more complex, as it is found in several branches of the tree. The prediction is mainly based on the number of devices and cars. Path 1: total_dispositivos (total electronic devices) <= 3.5 and num_auto (total number of cars) > 0.5. In this branch, the model focuses on the number of devices. If the household has 0.5 or fewer electronic devices (total_dispositivos <= 0.5), the model classifies it as Lower-middle stratum. Path 2: total_dispositivos (total electronic devices) > 3.5 and total_dispositivos <= 8.5. For this branch, the model considers the number of cars. If the household has 1.5 or fewer cars (num_car <= 1.5), it is likely to be classified as Lower-middle stratum. </li>
<li><strong>Upper-middle stratum:</strong> This stratum is also found in multiple branches, suggesting a pattern of higher asset ownership than the lower strata, but without reaching the highest level. Path 1: total_dispositivos (total electronic devices) <= 3.5, num_auto (total number of cars) > 0.5, and total_dispositivos > 0.5. In this branch, the model predicts Upper-middle stratum if the household has more than 0.5 cars, but still has 3.5 or fewer electronic devices. Path 2: total_dispositivos (total electronic devices) > 3.5, total_dispositivos <= 8.5, and num_auto (total number of cars) > 1.5. In this case, the model predicts Upper-middle stratum for households that have a higher number of devices (between 3.5 and 8.5) and more than 1.5 cars. </li>
<li><strong>High stratum: </strong>total_dispositivos (total electronic devices) > 3.5 and 8.5, and num_auto (total number of cars) > 2.5, the model predicts this stratum with high certainty.</li>
</p>

<p><a target="_blank" href="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/04_Decision%20Trees/04.%20Decision_Trees_Classifier_ENEIGH2024.ipynb"> Full code for this section</a></p>
<p><a href="#project-structure">Back to Project Structure</a></p>




<h2 id="segmentation-k-means">Random Forest Implementation 🚀</h2>

<p>After testing Decision Trees and obtaining promising results, we moved to Random Forest, an ensemble learning method that combines multiple decision trees to improve predictive performance and reduce overfitting. Random Forest is well-suited for our project because it can handle both categorical and numerical features, capture complex relationships, and provide more robust and stable predictions. The goal was to build a stronger classifier for socioeconomic strata by leveraging the power of multiple trees working together, thus achieving higher accuracy and generalizability. We will used the RandomForestClassifier module from sklearn and created 1,000 random trees. Finally, we used the same metrics as in the previous exercise, where we will look at metrics such as precision, recall, f1, andaccuracy:

<div align="center"><img src="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/05_Random%20Forest/metrics_randomforest.jpg" width="800" height="700" /></div>

According to the metrics generated with classification_report, the precission, recall, and f1-score of the algorithm increased, as did its overall precision. This means that Random Forest has been a good algorithm for developing the housing classifier according to the socioeconomic characteristics of the household. Finally, we determined which characteristics have been most important in creating our housing classifier. We used he feature_importances_ tool from RandomForestClassifier, which defined the importance of each feature in each tree based on the mean decrease in impurity (MDI). We also calculated the standard deviation of each attribute in each tree and finally generate a barplot to observe the 15 most important features:

<div align="center"><img src="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/05_Random%20Forest/feature_importance.png" width="800" height="700" /></div>

According to the graph, total_dispositivos, num_auto, ingreso_promedio, mat_pisos_3, and est_alim are feature with the highest MDI. This means all those variables are considered as important features for the random forest classifier. It seems the total number of electronic devices, number of cars, average income per household, type of floor, and average food expenditure define the household conditions according to this classifier.
</p>

<p><a target="_blank" href="https://github.com/victorve-l/Housing-well-being-classifier/blob/main/05_Random%20Forest/05.%20Random_Forest_ENEIGH2024.ipynb"> Full code for this section</a></p>
<p><a href="#project-structure">Back to Project Structure</a></p>

<h2 id="time-series-forecasting">Findings & Conclusions </h2>
<p>This study demonstrated the viability of using machine learning to predict socioeconomic strata based on housing and demographic data. By analyzing the ENIGH dataset, we built and evaluated several models, with the Random Forest algorithm providing the most insightful results.

The analysis revealed that material possessions, such as electronics and vehicles, and economic indicators like income and expenditure, are the most significant predictors of a household's well-being. The success of this classifier offers a promising avenue for streamlining the work of government agencies and researchers, allowing for a more dynamic and data-driven approach to understanding and addressing social inequality. Future work could involve incorporating more variables or deploying this model into a real-world application for public use.</p>

<p><a href="#project-structure">Back to Project Structure</a></p>
