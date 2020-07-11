# Debugging Python FastAPI apps in containers with Visual Studio code

*tl;dr: in this post I write about how a modern IDE like Visual Studio Code can ease debugging of a dockerized application [that I described before](https://davidefiocco.github.io/2020/06/27/streamlit-fastapi-ml-serving.html). The process is broken down in steps, but some basic level of familiarity with that project, working knowledge of Docker and debugging in Python is assumed.*

It makes sense to package machine learning-powered APIs within Docker containers: this practice allows collaborators (and your future self!) to run code on any machine (with enough resources!) capable of running Docker applications, without needing to worry about operating systems, language versions and distributions, library versions, etc. Docker also simplifies deployment and allows applications to be served easily in the cloud.

## Fixing ships in a bottle?

However, it can be cumbersome to perform code changes on a containerized application, because _outside the container_ the environment in general won't be properly configured to run it. To see how newly applied code changes affect behavior one would need to rebuild and rerun the container, and this can be slow and inefficient.

Working locally on code which eventually will be running in a container can feel a bit like building and _upgrading_ a model ship that needs to fit inside a bottle: one can build the ship (create the initial version of the code), put it in the bottle (build the container with the code) check if it fits inside nicely (run the code successfully in the container). However, if the ship doesn't fit (the code crashes) or if one wants to upgrade it (add new functionality to the code), they need to take it out and then inside the bottle again (modify the code, build and rerun the container), and again, and again...

![model ship](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Buddelschiff_2012_PD_06.JPG/1200px-Buddelschiff_2012_PD_06.JPG "Developing dockerized apps without proper tools can feel a bit like upgrading a ship in a bottle...")

Wouldn't it be nice to... magically enter the bottle and carry out all the desired changes to the ship while inside it?
The coding equivalent of this idea is _remote development in containers_. The idea is to perform code changes from within a container and when done with modifications, voilà, that's it!  
[This functionality is offered by Visual Studio Code](https://code.visualstudio.com/docs/remote/containers) (VS Code), which is a popular code editor developed by Microsoft.

## A concrete case

I recently published on GitHub a [simple example of dockerized API](https://github.com/davidefiocco/streamlit-fastapi-model-serving) based on FastAPI and streamlit. Very soon somebody noticed that it had a bug that make it struggle when handling some images and filed an [issue](https://github.com/davidefiocco/streamlit-fastapi-model-serving/issues/4). My "ship in the bottle" had a flaw that needed a fix.

How to analyse the buggy behavior?  
How to come up with the changes needed to fix it?  
How to add new features and improvements?  

Let's tackle these problems step by step below (in what follows I describe steps tailored [for my code example](https://github.com/davidefiocco/streamlit-fastapi-model-serving) and FastAPI, but with relatively minor changes one can debug a dockerized Python app in a similar way).

### Entering the "bottle"

To begin (check also [this screencast](/images/2020-07-11-opening-remote-container.png) to watch how the steps below play out in VS Code):

- make sure to have a recent version of VS Code installed (one can download it [here](https://code.visualstudio.com/download)) and that your machine can run Docker applications;
- clone the repo to be modified. To follow along, use commands below to load the "buggy" version of my code that needs improvement:

```bash
    git clone https://github.com/davidefiocco/streamlit-fastapi-model-serving
    git checkout 563ae1418d32ed36bda75bd4cb6c973b3d9d1cdb
```

- open the freshly-cloned `streamlit-fastapi-model-serving` folder in VS Code;
- install the [VS Code extension for development in containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and reload VS Code.
- click the green button in the bottom-left of VS Code and select the option `Remote-Containers: Reopen in Container` from the dropdown menu, selecting `From 'docker-compose.yml'` and `fastapi` as additional option (we choose `fastapi` as we assume is the one that needs a fix).

In this way, we've... entered "inside the bottle"! Our VS Code instance is running within the `fastapi` container, and the code can be modified and run _within a container_.

### Preparing to debug in the container

To understand what the problem may be, activating the VS Code Python debugger can be helpful.

To start working with the debugger with FastAPI, some prerequisites need to be satisfied (see [screencast](/images/2020-07-11-enable-debugging.png) for a visual walkthrough):

- install in the VS Code instance the [VS Code Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python), and reload VS Code;
- select a Python interpreter, which in this case is Python 3.7.7 in `/usr/local/bin`;
- from the "Run" menu of VS Code, select "Add configuration..." and "Python File";
- (as we're debugging a FastAPI API) follow the [FastAPI debugging page](https://fastapi.tiangolo.com/tutorial/debugging/), and perform the [few code changes](https://fastapi.tiangolo.com/tutorial/debugging/#call-uvicorn) needed to allow debugging;
- in the file explorer (left side of VS Code), edit the file `devcontainer.json` in the folder `.devcontainer` by uncommenting the `forwardPorts` attribute and set it further to `"forwardPorts": [8000]`. This allows a web browser running locally to reach the 8000 container port exposing the FastAPI documentation page;

Once done with the above, all is ready to start troubleshooting. To do so, focus the file `server.py` and press F5. The app should start executing in debug mode and the FastAPI-generated swagger interface should be reachable with a web browser at http://localhost:8000/docs soon after.

### Run in debugging mode, and rock the ship!

What's pretty cool is that it's possible to test the app by feeding it with images via the FastAPI-generated page, and step through the code with the VS Code Python debugger.

![debugging-fastapi](/images/2020-07-11-debugging-fastapi.png "Debugging the code in the container while firing requests via the FastAPI interface.")

To do so, I can open in VS Code `segmentation.py` (which contains the core Pytorch code), and place some breakpoints (the "red dots" on the left of the line numbers) at the lines where I want the Python execution to pause during debugging.

If I pass to the http://localhost:8000/docs frontend an "easy" image (say [this example](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21_thumb.jpg)), execution happily progresses from breakpoint to breakpoint while values of the variables are displayed on the left pane of VS Code. After all breakpoints have been stepped past, the API result is displayed in the browser window.

If I try a more "challenging" [high-res image](https://upload.wikimedia.org/wikipedia/commons/4/41/Left_side_of_Flying_Pigeon.jpg), the debugger execution hangs a very very long time at the line where the model is invoked: thanks to the debugger, I discovered that something fishy is going on there that needs a fix! The debugger can now be stopped with Shift+F5.

One delightful thing is that as VS code brings up the streamlit container as well from `docker-compose.yml`, I can trigger requests from the streamlit UI as well :)

![debugging-streamlit](/images/2020-07-11-debugging-streamlit.png "Debugging the code in the container while firing requests via the streamlit interface.")

### Coming up with a fix

Next, the idea is to come up with some code that can fix the problem. Proceeding by trial and error is allowed!

A procedure can be roughly as follows:

1. Do some code change (hopefully in the right direction!) and save the file;
2. Run the debugger for `server.py` (pressing F5);
3. Execute an API call via the http://localhost:8000/docs using a "problematic image";
4. Make the execution progress through the debugger and observe the outcome.

The steps above can be repeated until the API behaves as expected.  
Note that there's no container rebuilding needed in any of the above, and so experimenting with any wild idea is easy and fast.

The fix I ended up with (see the corresponding [PR on GitHub](https://github.com/davidefiocco/streamlit-fastapi-model-serving/pull/5/files)) consisted in a change in my `segmentation.py`:

```python
def get_segments(model, binary_image):

    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
```

writing a few lines of code to perform an image resize as a preprocessing step:

```python
def get_segments(model, binary_image, max_size=512):

    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size/width, max_size/height)
    resized_image = input_image.resize((int(input_image.width*resize_factor), int(input_image.height*resize_factor)))
```

and work with the newly-minted `resized_image` variable afterwards.

After having performed the change, I can run a [final debugging round](/images/2020-07-11-fix.png) and check that the code runs without major hiccups. The bug seems solved! So, after having configured `git` (if this wasn't done before) [I can commit the new version of the code](/images/2020-07-11-commit.png) and push it to its GitHub repository.

That's it. Note that we could keep upgrading the app indefinitely in the same fashion, by performing more and more changes and commits!

### Sail away

We've just scratched the surface of what can be done with tools like the Python debugger in containers. Feel free to give feedback via the Twitter handle below, or by [filing an issue](https://github.com/davidefiocco/davidefiocco.github.io/issues)!

Here are some reference links for more material:

- Python debugging in VS Code https://code.visualstudio.com/docs/python/debugging
- Remote development in containers https://code.visualstudio.com/docs/remote/containers
- Debugging FastAPI apps https://fastapi.tiangolo.com/tutorial/debugging/
