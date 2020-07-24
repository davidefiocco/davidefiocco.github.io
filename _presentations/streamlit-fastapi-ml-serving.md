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

### Machine learning can do cool stuff!

Consider _image segmentation_:

<section>
<img width="300" src="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21.jpg">

<img width="300" src="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21_class.png">
</section>

--

### We have trained a machine learning model... what now? 

We may want to serve it to users!

---

### We need a frontend...

![](../images/2020-06-27-streamlit.png)

--

### ...and a backend!

![](../images/2020-06-27-fastapi.png)

---

### Wouldn't it be nice to do it in pure Python?

---

### A solution using streamlit, FastAPI and Docker

- `streamlit`: to build the UI
- `fastapi`: to build the OpenAPI backend
- `docker-compose`: to orchestrate the two

---

### Idea was well received by the community

 <div class="tweet" data-src="https://twitter.com/monodavide/status/1276913357388382212?s=20" ></div>

--

![](../images/2020-06-27-github.png)

---

## You will learn about:

- a problem that can be solved with ML
- what `fastapi` and `streamlit` can do
- code of dockerized app with a frontend and backend

---

## References

- blogpost: https://davidefiocco.github.io/2020/06/27/streamlit-fastapi-ml-serving.html
- GitHub: https://github.com/davidefiocco/streamlit-fastapi-model-serving