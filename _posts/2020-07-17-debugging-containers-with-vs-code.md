# Debugging Python FastAPI apps in Docker containers with Visual Studio Code

*tl;dr: in this post I write about how a modern IDE like Visual Studio Code can ease debugging of a dockerized application [that I described before](https://davidefiocco.github.io/2020/06/27/streamlit-fastapi-ml-serving.html). The process is broken down in steps, but some basic level of familiarity with that project, working knowledge of Docker and debugging in Python is assumed.*

When developing machine learning-powered applications, encapsulating them in Docker containers offers clear advantages: 

- Docker allows apps to sit in **reproducible environments**: operating systems, language versions and distributions, library versions, are specified in the `Dockerfile` and other configuration files within the container (for example, Python packages are often specified in `requirements.txt`). The environment is thus configured as the container is built. This mechanism allows collaborators (and your future self!) to easily run code on any machine with enough resources capable of running Docker, and also simplifies quick deployment in the cloud.
- Docker allows to develop apps in **isolated environments**: each app can be developed in its very own "separate compartment", can run happily in isolation and not in a shared environment that may get [messier and messier and outright impossible to maintain over time](https://xkcd.com/1987/).

## How do you build a ship in a bottle?

There's a small catch though: it can be cumbersome to develop code of a containerized application, because _outside the container_ there won't be an environment able to run it. For example, to see how newly applied code changes affect the behavior of the app one would need to rebuild and rerun the container, and this can be slow and inefficient.

Working locally on code which eventually will be running in a container can feel a bit like building and _upgrading_ a model ship that needs to fit inside a bottle: one can build the ship (create the initial version of the code), put it in the bottle (build the container with the code) check if it fits inside nicely (run the code successfully in the container). However, if the ship doesn't fit (the code crashes) or needs upgrades (new functionality), one needs to take it out and then inside the bottle again (modify the code, build and rerun the container), and again, and again...

![model ship](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Buddelschiff_2012_PD_06.JPG/1200px-Buddelschiff_2012_PD_06.JPG "Developing dockerized apps without proper tools can feel a bit like building a ship in a bottle...")

Wouldn't it be nice to... magically enter the bottle and carry out all the desired changes to the ship while _inside the bottle_?
The coding equivalent of this idea is _remote development in containers_. Through development in containers one can perform code changes from within a container and when done with modifications, voilà, the updated application is ready to be run and deployed!  

In what follows, I'll describe an example of development in containers using [Visual Studio Code](https://code.visualstudio.com/) (VS Code). VS Code is a popular code editor developed by Microsoft which enables development in containers via [one of its _extensions_](https://code.visualstudio.com/docs/remote/containers) (note that some of the content here may become obsolete as new versions of the editor and extensions are released).

## A concrete case

I recently published on GitHub a [simple example of dockerized API](https://github.com/davidefiocco/streamlit-fastapi-model-serving) based on [FastAPI](https://fastapi.tiangolo.com/) and [streamlit](https://www.streamlit.io/). Very soon somebody noticed that the code was affected by a [bug](https://github.com/davidefiocco/streamlit-fastapi-model-serving/issues/4) that made it struggle when handling some images. My "ship in the bottle" had a flaw that needed a fix.

How to analyze the buggy behavior?  
How to come up with the changes needed to fix the bug?  
How to add new features and improvements?  

In what follows I tackle those questions for that code example and FastAPI specifically. However, with relatively minor changes, one can debug a dockerized Python app in a similar way.

### Entering the "bottle"

To begin (check also [this screencast](/images/2020-07-17-opening-remote-container.png) to watch how the steps below play out in VS Code):

- make sure to have a recent version of VS Code installed (downloadable [here](https://code.visualstudio.com/download)) on a machine that can run Docker applications;
- clone the repo to be modified. To follow along, use commands below to load the "buggy" version of my code that needs improvement:

```bash
    git clone https://github.com/davidefiocco/streamlit-fastapi-model-serving
    git checkout 563ae1418d32ed36bda75bd4cb6c973b3d9d1cdb
```

- open the freshly-cloned `streamlit-fastapi-model-serving` folder in VS Code;
- install the [VS Code extension for development in containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers);
- click the green button in the bottom-left of VS Code and select the option `Remote-Containers: Reopen in Container` from the dropdown menu, selecting `From 'docker-compose.yml'` and `fastapi` as additional option (I choose `fastapi` as I assume that's the one that needs a fix).

In this way, we've... entered "inside the bottle"! The VS Code instance is running within the `fastapi` container, and the code can be modified and run _within it_. Incidentally, by launching a terminal inside VS Code, one can also navigate the filesystem of the container.

### Preparing to debug in the container

To understand what problems the code has, the VS Code Python debugger can be a helpful tool.

To start working with the debugger with FastAPI, some prerequisites need to be satisfied (see [screencast](/images/2020-07-17-enable-debugging.png) for a visual walkthrough):

- install in the VS Code instance the [VS Code Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python), and reload VS Code;
- select a Python interpreter, which in this case is Python 3.7.7 in `/usr/local/bin`;
- from the "Run" menu of VS Code, select "Add configuration..." and "Python File";
- (as we're debugging a FastAPI API) follow the [FastAPI debugging page](https://fastapi.tiangolo.com/tutorial/debugging/), and perform the [few code changes](https://fastapi.tiangolo.com/tutorial/debugging/#call-uvicorn) needed to allow debugging;
- in the file explorer (left side of VS Code), edit the file `devcontainer.json` in the folder `.devcontainer` by uncommenting the `forwardPorts` attribute and set it to `"forwardPorts": [8000]`. This allows a web browser running locally to reach the 8000 container port exposing the FastAPI documentation page;

Once done with the above, troubleshooting can start! To do so, focus the file `server.py` and press F5. The app should start executing in debug mode and the FastAPI-generated swagger interface should be reachable with a web browser at http://localhost:8000/docs soon after.

### Run in debug mode, and rock the ship!

Note that it's possible to test the app by feeding it with images via the FastAPI-generated page, and step through the code with the VS Code Python debugger (pretty cool, uh?):

![debugging-fastapi](/images/2020-07-17-debugging-fastapi.png "Debugging the code in the container while firing requests via the FastAPI interface. Click [here](/images/2020-07-17-debugging-fastapi.png) for a larger version of the screencast.")

To do so I can place some breakpoints (the "red dots" on the left of the line numbers) at the lines where the Python execution should pause during debugging.
In my case, it makes sense to place breakpoints in `segmentation.py` (which contains the core PyTorch code), as that's the part of the code where hiccups may occur.

If I pass to the http://localhost:8000/docs FastAPI frontend an "easy" image (say [this example](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21_thumb.jpg)), execution happily progresses during debugging from breakpoint to breakpoint, while values of the variables are displayed on the left pane of VS Code. After the debugger has stepped past all breakpoints FastAPI displays output in the browser window. All looks OK!

If I try a more "challenging" [high-res image](https://upload.wikimedia.org/wikipedia/commons/4/41/Left_side_of_Flying_Pigeon.jpg) instead, the debugger execution hangs a very very long time at the line where the model is invoked: thanks to the debugger, I discovered that something fishy is going on there that needs a fix! The debugger can now be stopped with Shift+F5.

Incidentally, as VS code brings up the streamlit container as well from `docker-compose.yml`, I can trigger requests from the streamlit UI as well:

![debugging-streamlit](/images/2020-07-17-debugging-streamlit.png "Debugging the code in the container while firing requests via the streamlit interface. Click [here](/images/2020-07-17-debugging-streamlit.png) for a larger version of the screencast.")

### Coming up with a fix

I am thus all set to fix the problem, following a debugging procedure:

1. Do some code change (hopefully in the right direction!) and save the corresponding files;
2. Run the debugger for `server.py` (pressing F5);
3. Execute an API call via the http://localhost:8000/docs using a "problematic" image as input;
4. Make the execution progress through the debugger and observe the outcome.

The steps above can be repeated until the API behaves as expected. Note that there's no container rebuilding needed in any of the above, and so experimenting with any wild idea is pretty efficient.

One possible fix (see the corresponding [PR on GitHub](https://github.com/davidefiocco/streamlit-fastapi-model-serving/pull/5/files)) consists in changing `segmentation.py`:

```python
def get_segments(model, binary_image):

    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
```

adding a few lines of code to perform an image resize, so to create a smaller `resized_image` to be fed to the PyTorch model:

```python
def get_segments(model, binary_image, max_size=512):

    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size/width, max_size/height)
    resized_image = input_image.resize((int(input_image.width*resize_factor), int(input_image.height*resize_factor)))
```

After having performed the change, I can run a [final debugging round](/images/2020-07-17-fix.png) and check that the code runs without major hiccups. The bug seems indeed solved! So, after having configured `git` (if this wasn't done before) [I can commit the updated version of the code](/images/2020-07-17-commit.png) and push it to GitHub.

That's it! Note that one can keep upgrading the app indefinitely in the same fashion, by performing more and more changes and commits.

### Sail away

I've just scratched the surface of what can be done when developing and debugging Python APIs running on Docker with tools like VS Code. Feel free to send feedback via the Twitter handle below, or by [filing an issue](https://github.com/davidefiocco/davidefiocco.github.io/issues)! Here are some reference links for more material:

- Python development in VS Code as presented at Microsoft's Build May 2020 event by [@luumelo14](https://twitter.com/luumelo14): <https://channel9.msdn.com/Events/Build/2020/BOD100>
- Python debugging in VS Code: <https://code.visualstudio.com/docs/python/debugging>
- Remote development in containers: <https://code.visualstudio.com/docs/remote/containers>
- Debugging FastAPI apps: <https://fastapi.tiangolo.com/tutorial/debugging/>
