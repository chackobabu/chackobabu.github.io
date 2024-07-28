---
layout: default
title: Cycling Products Scraping
mathjax: true
---

#### **Scraping details of cycling products form selenium**

Importing the necessary libraries

```python
import pandas as pd
import json
import regex as re
from tqdm import tqdm
from IPython.display import clear_output
```
beautiful soup is important here, but its imported because I like to inspect the html element as a soup object at times. 
```python
from bs4 import BeautifulSoup
```


```python
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import ElementClickInterceptedException
import time
```


```python
driver = webdriver.Chrome()
driver.get("https://www.decathlon.in/")
```

defining a few helper functions to prevent repeition of complex selenuim commands.
```python
def wait_find(selector,d=driver):
    WebDriverWait(d, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
    return d.find_element(By.CSS_SELECTOR, selector)

def wait_findAll(selector,d=driver):
    WebDriverWait(d, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector)))
    return d.find_elements(By.CSS_SELECTOR, selector)

# this functions clears any output before printing, useful to check status of loops
def print_from_loop(exp):
    clear_output(wait=True)
    return print(exp)
```

The code below finds the searchbar on decathlon.in, enters cycles and presses enter. But this results might contain products that are not related to cycling too. eg. exercising cycles. In order to filter for cycles only, the code checks 'Cycles' under sports in the menu on the left side of the page. This ensures we only see cycling related projects. I had to put this in a loop becasue the required checkbox does not appear on the first go, at times.

```python
found_checkbox = False

for i in range(10):
    print_from_loop(f"iteration: {i+1}")
    search = wait_find("div.bg-grey-50.rounded-full")
    
    try:
        search.click()
    except ElementClickInterceptedException:
        continue
        
    search_bar = wait_find('input[type="text"]')
    search_bar.send_keys("cycle")
    search_bar.send_keys(Keys.ENTER)
    panels = wait_findAll(".overflow-scroll.uTYQCb.p-3 .ais-Panel")
    
    for each in panels:
        try:
            header = wait_find(".ais-Panel-header", d=each)
        except:
            continue
        
        if header.text == 'Sport':
            found_checkbox = True
    
    if found_checkbox:
        break
```

    iteration: 1
    

checks the box for cycles in the menu... 

```python
for p in panels:
    if wait_find(".ais-Panel-header", d=p).text == "Sport":
        sports = wait_findAll("li", d=p)

cycleRe = re.compile("(?i)cycle|cycling")

sel = False

for s in sports:

    if cycleRe.match(s.text):
    
        for i in range(10):
            try:
                s.click()
                sel = True
                break
            except ElementClickInterceptedException:
                continue
                
        if sel:
            break
```

checks the total number of products in the page from the text displayed... 

```python
string = wait_find(".text-center.my-2").text

tot = re.search(r"\d+", string).group()
tot = int(tot)
print(f"Total Products: {tot}")
```

    Total Products: 729
    

scrolls down till we load all the products...

```python
products = []
driver.execute_script(f"window.scrollTo(0, 0)")

while len(products) < tot:
    scroll = driver.execute_script("return window.scrollY + window.innerHeight")
    driver.execute_script(f"window.scrollTo(0, {scroll})")
    products = wait_findAll(".ais-InfiniteHits-item")
    print_from_loop(f"Total products found : {len(products)}")
```

    Total products found : 729
    
parsing urls from each product page...

```python
prod_urls = []
for p in tqdm(products):
    url = wait_find("a", d=p).get_attribute("href")
    prod_urls.append(url)
```

    100%|████████████████████████████████████████████████████████████████████████████████| 729/729 [01:37<00:00,  7.45it/s]
    

visiting each product page and getting the required information from there. 

```python
all_products = []

for url in tqdm(prod_urls):
    
    driver.get(url)
    deets = {}
    
    el = wait_find("script#__NEXT_DATA__")
    el = el.get_attribute('innerHTML')
    el1 = json.loads(el)['props']['initialState']

    p = json.loads(el1)['reducer']['productPage']['activeProduct']
    
    if (des := p.get('description')):
        deets['name'] = des.get('productName')
        deets['description'] =  des.get('descriptionShort')
        
        if (info := p['description'].get('informationConcept')):
            deets['gender'] = info.get('gender')
        
        if (benefits := p['description'].get('benefits')):
            deets['benefits'] = []
            for each in benefits:   
                deets['benefits'].append(each.get('name'))
                
    deets['mid'] = p.get('skuModelId')
    deets['brand'] = p.get('brand')
    deets['madeIn'] = p.get('madeIn')
    
    deets['breadcrumbs'] = []
    if (breadcrumbs := p.get('breadcrumbs')):
        for each in breadcrumbs:
            deets['breadcrumbs'].append(each.get('name'))
    
    if (price := p.get('priceForFront')):
        deets['price'] = price.get('finalPrice')
        deets['mrp'] = price.get('mrp')
        deets['discount'] = price.get('discountPercentage')
    
    if (review := p.get('review')):
        deets['averageRating'] = review.get('averageRating')
        deets['count'] = review.get('count')
        deets['countRecommended'] = review.get('countRecommended')
        deets['lastReviwed'] = review.get('lastReviwed')
    
    if p.get('productCategory'):
        deets['category'] = p['productCategory'].get('name')

    all_products.append(deets)
```

    100%|████████████████████████████████████████████████████████████████████████████████| 729/729 [15:16<00:00,  1.26s/it]
    


```python
df = pd.DataFrame(all_products)
```
```python
df
```


<pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>description</th>
      <th>gender</th>
      <th>benefits</th>
      <th>mid</th>
      <th>brand</th>
      <th>madeIn</th>
      <th>breadcrumbs</th>
      <th>price</th>
      <th>mrp</th>
      <th>discount</th>
      <th>averageRating</th>
      <th>count</th>
      <th>countRecommended</th>
      <th>lastReviwed</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hybrid Cycle Riverside 500 - Disc Brakes, Alum...</td>
      <td>Designed for leisure riding or occasional tour...</td>
      <td>MEN'S</td>
      <td>['versatility', 'ease of use', 'brake control'...</td>
      <td>8591163</td>
      <td>BTWIN</td>
      <td>India</td>
      <td>['All Sports', 'Cycling', 'Cycles']</td>
      <td>19999</td>
      <td>39999</td>
      <td>50</td>
      <td>3.79</td>
      <td>172.0</td>
      <td>114.0</td>
      <td>2022-01-30T00:01:29+01:00</td>
      <td>Hybrid Cycles</td>
    </tr>
    <tr>
      <th>1</th>
      <td>City Cycle Btwin My Bike - Steel Frame, Single...</td>
      <td>A single speed, simple cycle suited best for u...</td>
      <td>MEN'S</td>
      <td>['ease of use', 'efficiency', 'lifetime warran...</td>
      <td>8865743</td>
      <td>ROCKRIDER</td>
      <td>India</td>
      <td>['All Sports', 'Cycling', 'Cycles']</td>
      <td>6999</td>
      <td>8499</td>
      <td>0</td>
      <td>4.56</td>
      <td>192.0</td>
      <td>174.0</td>
      <td>20231006</td>
      <td>Mountain Bikes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mountain Bike Rockrider ST30 Grey - 7 Speed, M...</td>
      <td>The Rockrider ST30 Cycle is designed for leisu...</td>
      <td>MEN'S</td>
      <td>['cycling comfort', 'versatility', 'ease of us...</td>
      <td>8829659</td>
      <td>BTWIN</td>
      <td>India</td>
      <td>['All Sports', 'Cycling', 'Cycles']</td>
      <td>9999</td>
      <td>13999</td>
      <td>0</td>
      <td>4.18</td>
      <td>1281.0</td>
      <td>1064.0</td>
      <td>20231020</td>
      <td>Mountain Bikes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mountain Bike Rockrider ST20 High Frame - Stee...</td>
      <td>Designed for leisure cycling on off-road trail...</td>
      <td>MEN'S</td>
      <td>['cycling comfort', 'ease of use', 'efficiency...</td>
      <td>8866950</td>
      <td>BTWIN</td>
      <td>India</td>
      <td>['All Sports', 'Cycling', 'Cycles']</td>
      <td>7999</td>
      <td>11999</td>
      <td>0</td>
      <td>4.11</td>
      <td>1199.0</td>
      <td>981.0</td>
      <td>20240401</td>
      <td>Mountain Bikes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mountain Bike Rockrider ST20 Low Frame - Steel...</td>
      <td>Designed for leisure cycling on off-road trail...</td>
      <td>WOMEN'S</td>
      <td>['cycling comfort', 'ease of use', 'efficiency...</td>
      <td>8866951</td>
      <td>BTWIN</td>
      <td>India</td>
      <td>['All Sports', 'Cycling', 'Cycles']</td>
      <td>7999</td>
      <td>11999</td>
      <td>0</td>
      <td>4.13</td>
      <td>860.0</td>
      <td>688.0</td>
      <td>20240401</td>
      <td>Mountain Bikes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>724</th>
      <td>Rab Flatiron Badge Cap Heather</td>
      <td>Featuring a subtly contrasting vintage logo ba...</td>
      <td>MEN'S</td>
      <td>NaN</td>
      <td>afd7e3ca-08ac-4860-ad9c-5db158672928</td>
      <td>RAB</td>
      <td>NaN</td>
      <td>['Sports Accessories', 'Clothing accessories']</td>
      <td>1980</td>
      <td>1999</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Caps</td>
    </tr>
    <tr>
      <th>725</th>
      <td>Rab Flatiron Badge Cap Pine</td>
      <td>Featuring a subtly contrasting vintage logo ba...</td>
      <td>MEN'S</td>
      <td>NaN</td>
      <td>afd7e3ca-08ac-4860-ad9c-5db158672928</td>
      <td>RAB</td>
      <td>NaN</td>
      <td>['Sports Accessories', 'Clothing accessories']</td>
      <td>1980</td>
      <td>1999</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Caps</td>
    </tr>
    <tr>
      <th>726</th>
      <td>Rab Superfine Merino Beanie Red</td>
      <td>Ideal for cold weather layering or stand-alone...</td>
      <td>MEN'S</td>
      <td>NaN</td>
      <td>3cf859d9-42cf-4269-9b1b-c937406dfee3</td>
      <td>RAB</td>
      <td>NaN</td>
      <td>['Sports Accessories', 'Clothing accessories']</td>
      <td>2250</td>
      <td>2499</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Caps</td>
    </tr>
    <tr>
      <th>727</th>
      <td>500 COMPRESSION RUNNING SLEEVE</td>
      <td>Our design teams developed these compression r...</td>
      <td>NO GENDER</td>
      <td>['muscle support', None]</td>
      <td>8783975</td>
      <td>KIPRUN</td>
      <td>China</td>
      <td>['All Sports', 'Running']</td>
      <td>399</td>
      <td>549</td>
      <td>0</td>
      <td>4.51</td>
      <td>300.0</td>
      <td>273.0</td>
      <td>20240202</td>
      <td>New Arrival</td>
    </tr>
    <tr>
      <th>728</th>
      <td>Triban 100 700 Double-Walled Rear Free Wheel</td>
      <td>Designed for touring cyclists and racing cycli...</td>
      <td>NO GENDER</td>
      <td>['compatibility', 'durability', 'rolling effic...</td>
      <td>8528051</td>
      <td>TRIBAN</td>
      <td>France</td>
      <td>[]</td>
      <td>4999</td>
      <td>6999</td>
      <td>0</td>
      <td>4.17</td>
      <td>54.0</td>
      <td>46.0</td>
      <td>20230929</td>
      <td>Available Only Online</td>
    </tr>
  </tbody>
</table>
<p>729 rows × 16 columns</p>
</div>
</pre>
```python
df.to_excel("decathlon_cycling_products.xlsx", index=False)
```
<a href="\assets\data\decathlon_cycling_products.xlsx">if you would like to examine the table, you can download the it here</a>

**This is a work in progress, will update with the data visualisation part soon**