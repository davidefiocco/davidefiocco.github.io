---
theme : "night"
highlightTheme: "monokai"
slideNumber: false
title: "Machine learning model serving with streamlit and FastAPI"
author: "Davide Fiocco @monodavide"
logoImg: 
---

## Serving machine learning models with streamlit and FastAPI

Davide Fiocco  
@monodavide

---

### About me

- Started working with NumPy/SciPy in 2008
- (lots of data analyses in Python and R)
- Currently senior data scientist at Frontiers

---

<iframe class="stretch" data-src="https://www.frontiersin.org/" height=20%></iframe>

--

### Frontiers

- Open access publisher, 500+ employees
- In Lausanne, Madrid, London, Seattle, Beijing
- Processing 60k+ scientific articles / year
- Python used in ML and big data pipelines

---

### Machine learning is powerful!

--

### _Sentiment analysis_ (NLP):

- "_Granada is really beautiful!_" $\rightarrow$ positive 
- "_Too bad we're not there together_" $\rightarrow$ negative

--

### _Image segmentation_ (CV):

<section>
<img width="250" src="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21.jpg">

$\downarrow$

<img width="250" src="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21_class.png">
</section>

--

### Image segmentation in PyTorch:

Life is easy with pretrained models:

```python
import torch
model = torch.hub.load('pytorch/vision:v0.6.0',
                       'deeplabv3_resnet101',
                       pretrained=True)
model.eval()
```
after input preprocessing we get results with:

```python
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)
```

---

### We have a trained model... what now? 

Let's serve a prototype to users!

--

### We need a frontend...

<iframe class="stretch" data-src="http://localhost:8501" height=20%></iframe>

--

### ...and a backend

<iframe class="stretch" data-src="http://localhost:8000/docs" height=20% style="background: #ffffff;></iframe>


---

### Wouldn't it be nice to build it all in Python?

(and without writing much code?)

--

### A solution using streamlit, FastAPI and Docker

- `streamlit`: to build the frontend
- `fastapi`: to build the OpenAPI backend
- `docker-compose`: to orchestrate the two

---

## Streamlit

<iframe class="stretch" data-src="https://docs.streamlit.io/en/stable/" height=20%></iframe>

--

### streamlit features

- Design of simple UIs concisely
- Intuitive to use
- 10k+ starts on GitHub
- For comparisons, check https://plotly.com/comparing-dash-shiny-streamlit/

---

## FastAPI

<iframe class="stretch" data-src="https://fastapi.tiangolo.com/" height=20%></iframe>

--

### FastAPI features

- Concise syntax to add OpenAPI documentation
- Easy to use, similar to Flask
- Comprehensive documentation
- 20k+ stars on GitHub
- For comparisons, check https://fastapi.tiangolo.com/alternatives/

---

![Morpheus](https://i.kym-cdn.com/photos/images/original/001/186/986/75c.gif){:height="150%" width="150%"}

--

![Repo](../images/2020-06-27-github.png)

---

## Architecture

![Architecture](../images/containers-diagram.png =800x)

--

## Code tour

---

## Running the app (locally)

```bash
docker-compose build
docker-compose up
```

--

## Deploying the app on the web (on Heroku)

---

## Gracias! Thanks!

---

## References

- GitHub: https://github.com/davidefiocco/streamlit-fastapi-model-serving
- blogpost: https://davidefiocco.github.io/2020/06/27/streamlit-fastapi-ml-serving.html


