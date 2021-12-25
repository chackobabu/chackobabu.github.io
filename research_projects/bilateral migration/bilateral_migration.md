---
layout: default
title: "Gravity Model for Migration"
mathjax: true
---
#### **An analysis of gravity model for migrants of different skill levels**
<br>
What follows is an analysis I carried out as a private project, with the primary intention of mastering the various techniques of estimation. My main reference is <a href="https://www.imf.org/en/Publications/Spillover-Notes/Issues/2016/12/31/Impact-of-Migration-on-Income-Levels-in-Advanced-Economies-44343"  target="_blank" rel="noopener noreferrer">Jaumotte, F., Koloskova, K and Saxena S C. 2016</a>.

(If you would like to see the basic intuition behind a gravity model skip to [here](#1))

**The equation**

<div class="math-container">
$$
ln\text{MSH}_{odt} = \gamma_{0} + \gamma_{1}ln\text{pop}_{d1980} + \gamma_{2}ln\text{pop}_{o1980} + \gamma_{3}ln\text{MSH}_{od1980} + \gamma_{4}X_{ot} + \gamma_{5}Z_{od} + \gamma_{6}X_{ot}Z_{od} + \gamma_{t} + u_{odt} \tag{1}
$$
</div>

where,

$$MSH_{odt}$$ is the share of migrants from a given origin (o) at a given destination (d) at a given year (t)

$$\text{pop}_{d1980}$$ and $$\text{pop}_{o1980}$$ are the initial population size at destination and origin, respectively

$$MSH_{od1980}$$ is the initial share of migrants from a given origin at a given destination and captures network effects

$$X_{ot}$$ is a vector of push factors that consist of origin country growth, dummies for banking crises and civil wars, share of young population (25–34 years old), shares of population with tertiary and high school education, and dummy variable for being an EU member.

$$Z_{od}$$ is the vector of geography- and culture-based migration costs that include distance between the countries, dummies for contiguity, speaking a common ethnic language, shared past colonial ties, and membership in the EU. 

$$X_{ot}Z_{od}$$ is the interaction of push factors and costs.

$$δ_{t}$$ is the time fixed effects

$$u_{odt}$$ is the error term

**Estimates**

Below are the estimates for the gravity model. 

In the table, **'Total MSH'** is the share of total adult (age 25+) migrants from origin to destination among the adult population in destination, **'High Skilled MSH'** is the share of high skilled adult migrants from origin to destination among the high skilled adult population in destination. And **'Medium Skilled MSH'** and **'Low Skilled MSH'** can be similarly defined.

*(I have doubts about the shares. Should it be shares of migrants in the adult population of destination, or of origin. Here I have considered share of migrants in the adult population of destination. But I wonder if that variable can be fully explained by the independent variables. I will update on this soon.)*


Column (1): log-linear OLS regression with **log(Total MSH)** as dependendent variable.<br>
Column (2): Poisson pseudo-maximum likelihood regression with **log(Total MSH)** as dependendent variable.<br>
Column (3): Poisson pseudo-maximum likelihood regression with destination fixed effects with **log(Total MSH)** as dependendent variable.<br>
Column (4): Poisson pseudo-maximum likelihood regression with **log(High Skilled MSH)** as dependendent variable.<br>
Column (5): Poisson pseudo-maximum likelihood with **log(Medium Skilled MSH)** as dependendent variable.<br>
Column (6): Poisson pseudo-maximum likelihood with **log(Low MSH)** as dependendent variable.<br>

 <pre>
<table border="1" class="dataframe gravity">
  <thead>
    <tr>
      <th></th>
      <th>(1)</th>
      <th>(2)</th>
      <th>(3)</th>
      <th>(4)</th>
      <th>(5)</th>
      <th>(6)</th>
    </tr>
    <tr>
      <th></th>
      <th>ln(Total MSH) | OLS</th>
      <th>ln(Total MSH) | PPML</th>
      <th>ln(Total MSH) | PPML FE</th>
      <th>ln(High Skilled) | MSH PPML</th>
      <th>ln(Medium Skilled) | MSH PPML</th>
      <th>ln(Low Skilled) | MSH PPML</th>
    </tr>
    <tr>
      <th>variables</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ln(popl) at dest in 1980</th>
      <td>-0.071***</td>
      <td>-0.030***</td>
      <td>-0.012</td>
      <td>-0.068***</td>
      <td>-0.057***</td>
      <td>0.086***</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.012)</td>
      <td>(0.011)</td>
      <td>(0.036)</td>
      <td>(0.013)</td>
      <td>(0.018)</td>
      <td>(0.018)</td>
    </tr>
    <tr>
      <th>ln(popl) at orig in 1980</th>
      <td>0.132***</td>
      <td>0.045***</td>
      <td>0.038***</td>
      <td>0.112***</td>
      <td>0.079***</td>
      <td>-0.023</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.013)</td>
      <td>(0.014)</td>
      <td>(0.014)</td>
      <td>(0.016)</td>
      <td>(0.024)</td>
      <td>(0.017)</td>
    </tr>
    <tr>
      <th>ln(MSH) at dest in 1980</th>
      <td>0.806***</td>
      <td>0.635***</td>
      <td>0.674***</td>
      <td>0.485***</td>
      <td>0.493***</td>
      <td>0.618***</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.011)</td>
      <td>(0.013)</td>
      <td>(0.014)</td>
      <td>(0.014)</td>
      <td>(0.021)</td>
      <td>(0.017)</td>
    </tr>
    <tr>
      <th>ln(distance)</th>
      <td>-0.111***</td>
      <td>-0.069***</td>
      <td>-0.121***</td>
      <td>-0.064***</td>
      <td>-0.111***</td>
      <td>-0.022</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.021)</td>
      <td>(0.019)</td>
      <td>(0.023)</td>
      <td>(0.023)</td>
      <td>(0.037)</td>
      <td>(0.028)</td>
    </tr>
    <tr>
      <th>contiguity</th>
      <td>-0.013</td>
      <td>0.051</td>
      <td>-0.123</td>
      <td>-0.107</td>
      <td>0.186</td>
      <td>0.053</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.108)</td>
      <td>(0.088)</td>
      <td>(0.084)</td>
      <td>(0.096)</td>
      <td>(0.127)</td>
      <td>(0.111)</td>
    </tr>
    <tr>
      <th>common ethnic lang</th>
      <td>0.280***</td>
      <td>0.105**</td>
      <td>0.054</td>
      <td>0.375***</td>
      <td>0.242***</td>
      <td>-0.164***</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.054)</td>
      <td>(0.050)</td>
      <td>(0.054)</td>
      <td>(0.057)</td>
      <td>(0.082)</td>
      <td>(0.062)</td>
    </tr>
    <tr>
      <th>colony</th>
      <td>0.017</td>
      <td>-0.108*</td>
      <td>-0.154*</td>
      <td>0.020</td>
      <td>0.304***</td>
      <td>-0.510***</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.060)</td>
      <td>(0.060)</td>
      <td>(0.082)</td>
      <td>(0.069)</td>
      <td>(0.091)</td>
      <td>(0.086)</td>
    </tr>
    <tr>
      <th>EU member, orign &amp; dest</th>
      <td>3.992**</td>
      <td>3.795**</td>
      <td>3.059*</td>
      <td>-0.891</td>
      <td>1.440</td>
      <td>2.224</td>
    </tr>
    <tr>
      <th></th>
      <td>(1.758)</td>
      <td>(1.839)</td>
      <td>(1.850)</td>
      <td>(2.373)</td>
      <td>(3.293)</td>
      <td>(2.093)</td>
    </tr>
    <tr>
      <th>ln(real gdp pc) at origin-1980</th>
      <td>-0.075***</td>
      <td>-0.158***</td>
      <td>-0.177***</td>
      <td>-0.152***</td>
      <td>-0.126**</td>
      <td>-0.204***</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.026)</td>
      <td>(0.032)</td>
      <td>(0.031)</td>
      <td>(0.037)</td>
      <td>(0.051)</td>
      <td>(0.040)</td>
    </tr>
    <tr>
      <th>banking crisis at orign</th>
      <td>-0.110***</td>
      <td>-0.154***</td>
      <td>-0.164***</td>
      <td>-0.176***</td>
      <td>-0.199***</td>
      <td>-0.115**</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.026)</td>
      <td>(0.036)</td>
      <td>(0.035)</td>
      <td>(0.036)</td>
      <td>(0.051)</td>
      <td>(0.052)</td>
    </tr>
    <tr>
      <th>civil war at origin</th>
      <td>0.250***</td>
      <td>0.068</td>
      <td>0.090</td>
      <td>0.159**</td>
      <td>-0.031</td>
      <td>0.031</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.049)</td>
      <td>(0.056)</td>
      <td>(0.055)</td>
      <td>(0.073)</td>
      <td>(0.100)</td>
      <td>(0.099)</td>
    </tr>
    <tr>
      <th>% of young popl in origin (lagged)</th>
      <td>0.049***</td>
      <td>0.023**</td>
      <td>0.026**</td>
      <td>0.028***</td>
      <td>0.009</td>
      <td>0.034***</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.009)</td>
      <td>(0.010)</td>
      <td>(0.010)</td>
      <td>(0.009)</td>
      <td>(0.014)</td>
      <td>(0.013)</td>
    </tr>
    <tr>
      <th>ln(medium skilled popl share) at origin (lagged)</th>
      <td>0.015</td>
      <td>0.042</td>
      <td>0.051*</td>
      <td>0.141***</td>
      <td>0.063</td>
      <td>-0.046</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.023)</td>
      <td>(0.027)</td>
      <td>(0.027)</td>
      <td>(0.030)</td>
      <td>(0.048)</td>
      <td>(0.034)</td>
    </tr>
    <tr>
      <th>ln(high skilled popl share) at origin (lagged)</th>
      <td>0.044**</td>
      <td>0.021</td>
      <td>-0.004</td>
      <td>0.085**</td>
      <td>0.054</td>
      <td>0.029</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.020)</td>
      <td>(0.023)</td>
      <td>(0.021)</td>
      <td>(0.039)</td>
      <td>(0.043)</td>
      <td>(0.031)</td>
    </tr>
    <tr>
      <th>EU member, origin</th>
      <td>-0.194**</td>
      <td>-0.125**</td>
      <td>-0.181***</td>
      <td>-0.098</td>
      <td>0.030</td>
      <td>0.017</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.075)</td>
      <td>(0.063)</td>
      <td>(0.058)</td>
      <td>(0.072)</td>
      <td>(0.095)</td>
      <td>(0.074)</td>
    </tr>
    <tr>
      <th>cumul 5-year growth (lag)</th>
      <td>-0.597**</td>
      <td>0.331</td>
      <td>0.691</td>
      <td>-0.959**</td>
      <td>-1.598**</td>
      <td>0.854</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.277)</td>
      <td>(0.522)</td>
      <td>(0.492)</td>
      <td>(0.460)</td>
      <td>(0.703)</td>
      <td>(0.670)</td>
    </tr>
    <tr>
      <th>Constant</th>
      <td>0.640*</td>
      <td>1.207***</td>
      <td>1.851***</td>
      <td>0.511</td>
      <td>0.794</td>
      <td>2.221***</td>
    </tr>
    <tr>
      <th></th>
      <td>(0.353)</td>
      <td>(0.432)</td>
      <td>(0.549)</td>
      <td>(0.488)</td>
      <td>(0.755)</td>
      <td>(0.524)</td>
    </tr>
    <tr>
      <th></th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>Observations</th>
      <td>9,710</td>
      <td>9,764</td>
      <td>9,764</td>
      <td>9,764</td>
      <td>9,764</td>
      <td>9,764</td>
    </tr>
    <tr>
      <th>R-squared</th>
      <td>0.871</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>Year FE</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>Interaction Terms</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>Destination FE</th>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>Robust standard errors in parentheses</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</pre>
<a href="/research_projects/bilateral migration/reg_table.xlsx">if you would like to examine the table closely, you can download the sheet here</a>

<br>
The estimates seem to more or less conform to the basic intuition. Adverse socio-economic conditions at the origin (eg civil war) and lower migration costs (eg distance) increase the share of migrants. There is also strong evidence for network effects (initial share of migrants). There are variations in importance of push factors and cost factors between differently-skilled migrants. 

(detailed analysis need to be done)

The dataset that I put together for the purpose can be downloaded <a href="/research_projects/bilateral migration/all_vars_gravity_migration.xlsx">here</a>.  The python notebook (which has codes that put together the data) and the stata do file (which contains codes for the estimation) can be provided to those interested. 

The various data sources are:
<ul>
<li>Education level of immigrants: <a href="https://www.iab.de/en/ueberblick.aspx" target="_blank" rel="noopener noreferrer"> Institute for Employment Research dataset</a></li>


<li>Educational attainment of population: <a href="https://barrolee.github.io/BarroLeeDataSet/" target="_blank" rel="noopener noreferrer">Barro-Lee data set</a></li>

<li>Population figures: <a href="https://population.un.org/wpp/" target="_blank" rel="noopener noreferrer">UN Population projections</a></li>

<li>real GDP per capita and growth rate: <a href="https://www.rug.nl/ggdc/productivity/pwt/?lang=en" target="_blank" rel="noopener noreferrer">Penn World Tables version 10.0</a></li>

<li>Gravity vairables (distance, contiguity, colony, common ethnic language): <a href="http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele.asp" target="_blank" rel="noopener noreferrer">Centre d’Études Prospectives et d’Informations Internationales database</a></li>

<li>Banking crisis: <a href="https://www.imf.org/en/Publications/WP/Issues/2018/09/14/Systemic-Banking-Crises-Revisited-46232" target="_blank" rel="noopener noreferrer">Laeven and Valencia’s (2012) data set</a></li>

<li>Civil war: <a href="https://www.systemicpeace.org/inscrdata.html" target="_blank" rel="noopener noreferrer">Center For Systemic Peace data</a></li>

</ul>

There is much more work to be done on this. Any comments/suggestions will be appreciated.

<a id=1></a>

**Basic intuition behind gravity model for migration**

At the very basic level gravity equation for migration can be written as:

$$
\text{log Migrants}_{od} = c + b_{1}\text{log GDP}_{o} + b_{2}\text{log GDP}_{d} - b_{3}\text{log}(\text{distance}_{od}) +e_{od} \tag{2}
$$
 
where,

$$\text{Migrants}_{od}$$ is the number of migrants from origin country o to destination country d
 
$$\text{GDP}_{o}$$ is the GDP of origin country o

$$\text{GDP}_{d}$$ is the GDP of destination country d
 
$$\text{distance}_{od}$$ is the distance between the two countries o and d

$$e_{ij}$$ is a random error term, $$c$$ term is a regression constant, and the $$b$$ terms are coefficients to be estimated.

The name “gravity” comes from the fact that the "nonlinear" form of equation 2 resembles Newton’s law of gravity: number of migrants between two nations is directly proportional to the destination and origin countries’ economic “mass” (GDP), and inversely proportional to the distance between them.

These relationships can indeed be observed from data, as shown below. 

The figures (of migrants, distance and GDP) are averaged over years for each of the country pair.

**Scatter plot and line of best fit for number of migrants versus product of GDPs.**
  <img src="/assets/img/research projects/bilateral migration gravity/mig_gdp.png" width="70%"> 

The relationship between migration and product of GDPs is positive, as the basic intuition suggests

**Scatter plot and line of best fit for number of migrants versus distance.**
  <img src="/assets/img/research projects/bilateral migration gravity/mig_dist.png" width="70%">

The relationship between migration and distance is negative, as the basic intuition suggests.

If anybody wishes to download the data and try the graphs by themselves, its <a href="/research_projects/bilateral migration/gravity_intuition.xlsx">here</a>. 

The GDP variables were downloaded from <a href="https://data.worldbank.org/indicator/NY.GDP.MKTP.KD" target="_blank" rel="noopener noreferrer">World Bank Database</a>. The rest of the variables were downloaded for the excercise above, from the source mentioned in that section.
