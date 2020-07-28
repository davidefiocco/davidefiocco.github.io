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

### Machine learning is cool!

Consider _sentiment analysis_ in NLP:

- _This talk is already boring_ $\rightarrow$ negative
- _This talk looks promising!_ $\rightarrow$ positive 

--

### Machine learning is cool!

Consider _image segmentation_ in CV:

<section>
<img width="300" src="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21.jpg">

<img width="300" src="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21_class.png">
</section>

--

### Example: DeepLabV3 in PyTorch

---

### We have a trained model... what now? 

Let's serve a prototype to users!

--

### We need a frontend...

<iframe class="stretch" data-src="http://localhost:8501" height=20%></iframe>

--

### ...and a backend!

<iframe class="stretch" data-src="http://localhost:8000/docs" height=20%></iframe>

---

### Wouldn't it be nice to build it all in Python?

(and without writing much code?)

--

### A solution using streamlit, FastAPI and Docker

- `streamlit`: to build the frontend
- `fastapi`: to build the OpenAPI backend
- `docker-compose`: to orchestrate the two

--

![Repo](../images/2020-06-27-github.png)

---

## Streamlit

<div class="tweet" data-src="https://twitter.com/streamlit/status/1272892481470857232" height=80%></div>

--

### Streamlit features

---

## FastAPI

<iframe class="stretch" data-src="https://fastapi.tiangolo.com/" height=20%></iframe>

--

### FastAPI features

---

![Morpheus](https://i.kym-cdn.com/photos/images/original/001/186/986/75c.gif){:height="150%" width="150%"}

---

## Architecture

--

![](../images/containers-diagram.png){:width="80%"} <!-- .element: style="float: right; width: 20%" -->

```yaml
version: '3'

services:

  streamlit:
    build: streamlit/
    depends_on:
      - fastapi
    ports: 
        - 8501:8501
    networks:
      - deploy_network
    container_name: streamlit

  fastapi:
    build: fastapi/
    ports: 
      - 8000:8000
    networks:
      - deploy_network
    container_name: fastapi

networks:
  deploy_network:
    driver: bridge

``` 
<!-- .element: style="width: 50%" -->

--

Diagram with same container architecture and file structure

---

## Backend

--

PyTorch code and tutorial reference

--

Page with FastAPI interface and code

---

## Frontend

--

Page with streamlit interface and code

---

## Running the app

--

### Running the app locally

--

### Deploying the app on the web (on Heroku)

---

## References

- blogpost: https://davidefiocco.github.io/2020/06/27/streamlit-fastapi-ml-serving.html
- GitHub: https://github.com/davidefiocco/streamlit-fastapi-model-serving

---

## Gracias! Thanks!