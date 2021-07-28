# This dockerizable flask+gunicorn app can be used as template to generate a complying challenge submission.

If any problems or questions arise, please contact us.

The goal of this architecture is to make it easy for you to deploy and submit your model and adaptation procedures. To evaluate your algorithms, we will access them via REST API calls.

We expect most teams to use python. An adaptation to any other language is possible of course, as long as the REST API stays the same. Run the example evaluation script to test, see below.

----

## Basics:
  - Docker as virtualization platform
  - Flask as web framework
  - Gunicorn as web server
  - REST API to communicate with the dockerized application

If you're new to these technologies, don't worry. Basically you only have to call your own code from one file, `views.py`. The rest should work out-of-the-box.

----

This is how we suggest you approach this code:
## 1. Look at the api_challenge_eval.yaml file
This file defines the REST API call. There is only one endpoint: `/predict`.

To visualize this, you can either use an openapi/swagger IDE extension or paste the file into https://editor.swagger.io/.

Each call contains labelled adaptation equations and unlabelled validation equations.

The file also defines what is expected to be returned: an ordered list of equation hypotheses.

All this is already taken care of in this flask/gunicorn docker template.

## 2. Get the template running without adding your own code
There are two ways of running this server. Either with the flask development server or dockerized with gunicorn as production server.

In either case, we recommend installing the requirements to your (virtual) python environment. See `requirements.txt` and `eval_submission.py` for what to install via pip.

### Flask Development Server (development only)
The flask server is usually used during development only and should be replaced by the gunicorn+docker solution below before submitting your algorithms.

`cd` to flask_app/

optional: add an environment variable to enter debug mode, `$env:FLASK_ENV = "development"` (syntax varies depending on terminal and os)

`python -m flask run -p 8080` runs the flask app on port 8080. You should see this output:
```
* Environment: development
* Debug mode: on
* Restarting with stat
* Debugger is active!
* Debugger PIN: xxx
* Running on http://127.0.0.1:8080/ (Press CTRL+C to quit)
```

You can now run the script that demonstrates the evaluation procedure (see section 3) or get the gunicorn/docker version running first.

### Docker + Gunicorn (submission format)
First, install Docker: https://docs.docker.com/get-docker/

`cd` to flask_app/

To build a Docker image, we need a Dockerfile, gunicorn.sh (the entrypoint) and requirements.txt. Feel free to inspect those files in detail.

Build the docker image with `docker build . -t ubicomp21_teamname:1.0.0`

In case you make use of nvidia GPU support, run `docker build . -f Dockerfile-gpu -t ubicomp21_teamname:1.0.0`, also for all future docker builds. The computer to run this on later needs to have nvidia-smi installed.

The output should be similar to:
```
[+] Building 2.7s (10/10) FINISHED
 => [internal] load build definition from Dockerfile    0.1s 
 => => transferring dockerfile: 237B    0.0s 
 => [internal] load .dockerignore   0.1s 
 => => transferring context: 2B 0.0s 
 => [internal] load metadata for docker.io/library/python:3.8-slim  1.8s 
 => [1/5] FROM docker.io/library/python:3.8-slim@sha256:1156cbb1f6a7660dcce3e2f3906a149427fbee71aea0b49953bccf0cc7a3bcaa    0.0s 
 => [internal] load build context   0.1s 
 => => transferring context: 4.05kB 0.0s 
 => CACHED [2/5] COPY requirements.txt  0.0s 
 => CACHED [3/5] RUN pip3 install -r /requirements.txt  0.0s 
 => [4/5] COPY . /app   0.2s 
 => [5/5] WORKDIR /app  0.2s 
 => exporting to image  0.2s 
 => => exporting layers 0.1s 
 => => writing image sha256:62b32352506cb3ff378b768899c6909e9b776568e56f10665539a927b0f5657b    0.0s 
 => => naming to docker.io/library/ubicomp21_teamname:1.0.0      
```

Run the docker file we just built with `docker run -p 8080:80 --name ubicomp21_teamname ubicomp21_teamname:1.0.0`. We map the ports and give the container a name.

In case you make use of nvidia GPU support, run `docker run -p 8080:80 --gpus all --name ubicomp21_teamname ubicomp21_teamname:1.0.0`, also for all future docker builds. This won't work if your computer does not have nvidia-smi available.

The output should be similar to:
```
Hello Ubicomp Challenge 2021 World!
[2021-05-21 07:21:51 +0000] [7] [DEBUG] Current configuration:
  config: ./gunicorn.conf.py
  wsgi_app: None
  bind: ['0.0.0.0:8080']
  [...]
[2021-05-21 07:21:51 +0000] [7] [INFO] Starting gunicorn 20.1.0
[2021-05-21 07:21:51 +0000] [7] [DEBUG] Arbiter booted
[2021-05-21 07:21:51 +0000] [7] [INFO] Listening at: http://0.0.0.0:8080 (7)
[2021-05-21 07:21:51 +0000] [7] [INFO] Using worker: sync
[2021-05-21 07:21:51 +0000] [9] [INFO] Booting worker with pid: 9
[2021-05-21 07:21:51 +0000] [7] [DEBUG] 1 workers
```

## 3. Run the script that demonstrates the evaluation procedure
Great, we got the server running. Let's run the script that simulates the evaluation STABILO will do with your submitted algorithms.

**Note:** You can now also use the "Try it out" feature of your local openapi/swagger preview to make the server respond to some mock data.

Open the file `eval_submission.py` and change the `folder` path so that it points to the training data. The secret validation data will have the same folder and file format.
To test on more than one person, remove the `break` in the read_dataset() method.

Run the script. The output should be similar to this one:
```
Collecting data for person 0...
-- Recording 0: 10 equations
-- Recording 1: 65 equations
-- Recording 2: 186 equations
-- Recording 3: 180 equations
Summary:
Persons: 1
Overall equations: 441

person nr. 0 - status code: 201
Hypo: 123+123=246, Label: 2646-29, Lev-Distance:9
Hypo: 123+123=246, Label: 80+67+0+77, Lev-Distance:10
Hypo: 123+123=246, Label: 9-1007=3, Lev-Distance:10
[...]
Hypo: 123+123=246, Label: 8-7+83·83, Lev-Distance:9
RESULTS:
Overall Levenshtein Distance: 9.662844036697248
Overall Word accuracy: 0.0
```

That's it: The average Levenshtein distance of the template server is 9.66 per equation. The word accuracy is evaluated but doesn't play a role when determining the winning teams.

## 4. Add your own your adaptation and prediction code
In `app/views.py`, the API call to `/predict` is being processed. The json data are parsed and transformed to `Pandas.DataFrame` objects. You may of course change that.

From views.py, you should call your adaptation and prediction code. See the big comments within the code.

Make sure to add your requirements to the `requirements.txt` file to make sure they are automatically installed in your docker container.

## 5. Rebuild the docker file, run, evaluate yourself, write to image file, submit.
To encorporate your own algorithms in the docker file, rebuild and run the docker image as shown above.

Use the `eval_submission.py` script to make sure your dockerized flask/gunicorn server is working as expected.

To submit the built docker image, use `docker save -o /.../ubicomp21_teamname.tar ubicomp21_teamname:1.0.0` to save the image file.
To check if your generated tar file works, try `docker load --input /.../ubicomp21_teamname.tar` followed by the usual `docker run -p 8080:80 --name ubicomp21_teamname ubicomp21_teamname:1.0.0` and try out the evaluation script again.


Send us the generated .tar file, all your source code along with an open source license, documentation and a short written report (pdf, 3-6 pages) to evaluate your algorithms on unseen writers.

----

Have fun and thanks for participating!

In case of any questions, don't hesitate to contact us via stabilodigital.com/contact.