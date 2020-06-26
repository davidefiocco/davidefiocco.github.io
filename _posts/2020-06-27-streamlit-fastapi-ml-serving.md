# Machine learning model serving in Python using FastAPI and streamlit

In my current job I train machine learning models. When experiments show that one of these models can solve some need of the company, we sometimes serve it to users in the form of a "prototype" deployed on internal servers. While such a prototype may not be production-ready yet, it can be useful to show to users strengths and weeknesses of the proposed solution and get feedback so to release better iterations.

Such a prototype needs to have:

1. a frontend (a user interface aka UI), so that users can interact with it;
2. a backend with API documentation, so that it can process a lot of requests in bulk and moved easily to production and integrated with other applications later on.

Also, it'd be nice to create these easily, quickly and concisely, so that more attention and time can be devoted to better data and model development!

In the recent past I have dabbled in HTML and Javascript to create UIs, and used Flask to create the underlying backend services. This did the job, but:

- I could just create very simple UIs (using [bootstrap](https://getbootstrap.com/) and [jQuery](https://jquery.com/)), but had to bug my colleagues to make them functional and not totally ugly.
- My Flask API endpoints were very simple and they didn't have API documentation. They also served results using the server built-in Flask which is [not suitable for production](https://flask.palletsprojects.com/en/1.1.x/deploying/).

## What if both frontend and backend could be easily built with (little) Python?

You may already have heard of FastAPI and streamlit, two Python libraries that lately are getting quite some attention in the applied ML community.

[FastAPI](https://fastapi.tiangolo.com/) is [gaining popularity](https://twitter.com/honnibal/status/1272513991101775872) among Python frameworks. It is thoroughly documentated, allows to code APIs following [OpenAPI specifications](https://en.wikipedia.org/wiki/OpenAPI_Specification) and can use `uvicorn` behind the scenes, allowing to make it "good enough" for some production use. Its syntax is also similar to that of Flask, so that its easy to switch to it if you have used Flask before.

[streamlit](https://www.streamlit.io/) is [getting traction](https://twitter.com/streamlit/status/1272892481470857232?s=20) as well. It allows to create pretty complex UIs in pure Python. It can be used to serve ML models without further ado, but (as of today) [you can't build REST endpoints with it](https://github.com/streamlit/streamlit/issues/439).

So why not combine the two, and get the best of both worlds?

## A simple "full-stack" application: image semantic segmentation with DeepLabV3

As an example, let's take *image segmentation*, which is the task of assigning to each pixel of a given image to a category (for a primer on image segmentation, check out the [fast.ai course](https://course.fast.ai/videos/?lesson=3)).  
Semantic segmentation can be done using a model pre-trained on images labeled using predefined list of categories. An example in this sense is [DeepLabV3](https://arxiv.org/pdf/1706.05587.pdf)) these have been already [implemented in PyTorch](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/).  
How can we serve those in an a app with a streamlit frontend and FastAPI backend?

One possibility is to have two services deployed in two Docker containers, orchestrated with `docker-compose`:

```yml
version: '3'

services:
  fastapi:
    build: fastapi/
    ports: 
      - 8000:8000
    networks:
      - deploy_network
    container_name: fastapi

  streamlit:
    build: streamlit/
    depends_on:
      - fastapi
    ports: 
        - 8501:8501
    networks:
      - deploy_network
    container_name: streamlit

networks:
  deploy_network:
    driver: bridge
```

The `streamlit` service serves a UI that calls (using the `requests` package) the endpoint exposed by the `fastapi` service, while UI elements (text, fileupload, buttons, display of results), are declared with calls to `streamlit`:

```python
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PIL import Image
import io

st.title('DeepLabV3 image segmentation')

# fastapi endpoint
url = 'http://fastapi:8000'
endpoint = '/segmentation'

st.write('''Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8000/docs` for FastAPI documentation.''') # description and instructions

image = st.file_uploader('insert image')  # image upload widget


def process(image, server_url: str):

    m = MultipartEncoder(
        fields={'file': ('filename', image, 'image/jpeg')}
        )

    r = requests.post(server_url,
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000)

    return r


if st.button('Get segmentation map'):
    segments = process(image, url+endpoint)
    segmented_image = Image.open(io.BytesIO(segments.content)).convert('RGB')
    st.image([image, segmented_image], width=300) # output dyptich
```

The FastAPI backend calls some methods from an auxiliary module `segmentation.py`, and implements a `/segmentation` endpoint giving an image in output (handling images with FastAPI can be a bit tricky but [it can definitely be done](https://stackoverflow.com/a/55905051/4240413)):


```python
from fastapi import FastAPI, File
import tempfile
from starlette.responses import FileResponse
from segmentation import get_segmentator, get_segments

model = get_segmentator()

app = FastAPI(title="DeepLabV3 image segmentation",
              description='''Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch. 
                           Visit this URL at port 8501 for the streamlit interface.''',
              version="0.1.0",
              )


@app.post("/segmentation")
async def get_segmentation_map(file: bytes = File(...)):
    '''Get segmentation maps from image file'''
    segmented_image = get_segments(model, file)
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".png", delete=False) as outfile:
        segmented_image.save(outfile)
        return FileResponse(outfile.name, media_type="image/png")
```

One just needs to add Dockerfiles, `pip` requirements and the core Pytorch code (stealing from the [official tutorial](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)) to come up with a [complete solution](https://github.com/davidefiocco/streamlit-fastapi-model-serving/).

Note that we're dealing with images in this example, but we could definitely modify it to use other kind of data in input and output!

To test the application locally one can simply execute in a command line

```bash
    docker-compose build
    docker-compose up
```

and then visit http://localhost:8000/docs with a web browser to interact with the FastAPI-generated documentation, which should look a bit like this:

![](/images/2020-06-27-fastapi.png "FastAPI documentation")

The streamlit-generated page can be visited at http://localhost:8501, and after uploading an image example and pressing the button, you should see the original image and the semantic segmentation generated by the model:

![](/images/2020-06-27-streamlit.png "streamlit UI")

One can try to run the model on more images to experiment with its strengths and weaknesses (classes used in the segmentation are those of the [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#introduction)).

With essentially no changes, it's then possible to deploy the application on the web (e.g. with [Heroku](http://heroku.com/)). Note that quite some elements will be still missing to call it a full-blown production application: secure authentication (which can be enabled via FastAPI), dealing with concurrency under heavy load, monitoring, logging,... some of which could be dealt with in another post!  

Still, it's impressive how not-so-basic data products can be created with so little Python code!