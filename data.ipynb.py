{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting git+https://github.com/google-research/tensorflow_constrained_optimization\n",
      "  Cloning https://github.com/google-research/tensorflow_constrained_optimization to /private/var/folders/j3/rkw2h6813z3fs6ngq8579q540000gn/T/pip-req-build-l8y14b6s\n",
      "  Running command git clone -q https://github.com/google-research/tensorflow_constrained_optimization /private/var/folders/j3/rkw2h6813z3fs6ngq8579q540000gn/T/pip-req-build-l8y14b6s\n",
      "Requirement already satisfied (use --upgrade to upgrade): tfco-nightly==0.3.dev20200922 from git+https://github.com/google-research/tensorflow_constrained_optimization in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages\n",
      "Requirement already satisfied: numpy in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tfco-nightly==0.3.dev20200922) (1.17.2)\n",
      "Requirement already satisfied: scipy in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tfco-nightly==0.3.dev20200922) (1.4.1)\n",
      "Requirement already satisfied: six in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tfco-nightly==0.3.dev20200922) (1.12.0)\n",
      "Requirement already satisfied: tensorflow>=1.14 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tfco-nightly==0.3.dev20200922) (2.3.0)\n",
      "Requirement already satisfied: h5py<2.11.0,>=2.10.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (2.10.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (1.1.0)\n",
      "Requirement already satisfied: wrapt>=1.11.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (1.11.2)\n",
      "Requirement already satisfied: grpcio>=1.8.6 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (1.32.0)\n",
      "Requirement already satisfied: keras-preprocessing<1.2,>=1.1.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (1.1.2)\n",
      "Requirement already satisfied: absl-py>=0.7.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (0.8.0)\n",
      "Requirement already satisfied: tensorboard<3,>=2.3.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (2.3.0)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (3.1.0)\n",
      "Requirement already satisfied: wheel>=0.26 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (0.33.6)\n",
      "Requirement already satisfied: gast==0.3.3 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (0.3.3)\n",
      "Requirement already satisfied: tensorflow-estimator<2.4.0,>=2.3.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (2.3.0)\n",
      "Requirement already satisfied: astunparse==1.6.3 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (1.6.3)\n",
      "Requirement already satisfied: google-pasta>=0.1.8 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (0.1.8)\n",
      "Requirement already satisfied: protobuf>=3.9.2 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (3.13.0)\n",
      "Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (1.7.0)\n",
      "Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (0.4.1)\n",
      "Requirement already satisfied: werkzeug>=0.11.15 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (0.16.0)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (2.24.0)\n",
      "Requirement already satisfied: google-auth<2,>=1.6.3 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (1.21.2)\n",
      "Requirement already satisfied: setuptools>=41.0.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (41.4.0)\n",
      "Requirement already satisfied: markdown>=2.6.8 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (3.1.1)\n",
      "Requirement already satisfied: requests-oauthlib>=0.7.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (1.3.0)\n",
      "Requirement already satisfied: chardet<4,>=3.0.2 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (3.0.4)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (2019.9.11)\n",
      "Requirement already satisfied: idna<3,>=2.5 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (2.8)\n",
      "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (1.24.2)\n",
      "Requirement already satisfied: rsa<5,>=3.1.4; python_version >= \"3.5\" in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (4.6)\n",
      "Requirement already satisfied: pyasn1-modules>=0.2.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (0.2.8)\n",
      "Requirement already satisfied: cachetools<5.0,>=2.0.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (4.1.1)\n",
      "Requirement already satisfied: oauthlib>=3.0.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (3.1.0)\n",
      "Requirement already satisfied: pyasn1>=0.1.3 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from rsa<5,>=3.1.4; python_version >= \"3.5\"->google-auth<2,>=1.6.3->tensorboard<3,>=2.3.0->tensorflow>=1.14->tfco-nightly==0.3.dev20200922) (0.4.8)\n",
      "Building wheels for collected packages: tfco-nightly\n",
      "  Building wheel for tfco-nightly (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for tfco-nightly: filename=tfco_nightly-0.3.dev20200922-cp37-none-any.whl size=178894 sha256=09750ac157639f248d28bac7565df64c2954063442e37a3dc859542049082b25\n",
      "  Stored in directory: /private/var/folders/j3/rkw2h6813z3fs6ngq8579q540000gn/T/pip-ephem-wheel-cache-18h1iy9b/wheels/c9/b3/c3/78e0691949466af462380554286105216cd95a9ae7cf08ee78\n",
      "Successfully built tfco-nightly\n",
      "Requirement already satisfied: fairness-indicators in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (0.24.0)\n",
      "Requirement already satisfied: absl-py==0.8.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (0.8.0)\n",
      "Requirement already satisfied: pyarrow<0.17,>=0.16 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (0.16.0)\n",
      "Requirement already satisfied: apache-beam<3,>=2.20 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (2.24.0)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: avro-python3==1.9.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (1.9.1)\n",
      "Requirement already satisfied: pyzmq==17.0.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (17.0.0)\n",
      "Requirement already satisfied: tensorflow-model-analysis<0.25,>=0.24 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from fairness-indicators) (0.24.2)\n",
      "Requirement already satisfied: witwidget<2,>=1.4.4 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from fairness-indicators) (1.7.0)\n",
      "Requirement already satisfied: tensorflow-data-validation<0.25,>=0.24 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from fairness-indicators) (0.24.0)\n",
      "Requirement already satisfied: tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from fairness-indicators) (2.3.0)\n",
      "Requirement already satisfied: six in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from absl-py==0.8.0) (1.12.0)\n",
      "Requirement already satisfied: numpy>=1.14 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from pyarrow<0.17,>=0.16) (1.17.2)\n",
      "Requirement already satisfied: typing-extensions<3.8.0,>=3.7.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (3.7.4.3)\n",
      "Requirement already satisfied: oauth2client<4,>=2.0.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (3.0.0)\n",
      "Requirement already satisfied: python-dateutil<3,>=2.8.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (2.8.0)\n",
      "Requirement already satisfied: hdfs<3.0.0,>=2.1.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (2.5.8)\n",
      "Requirement already satisfied: crcmod<2.0,>=1.7 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (1.7)\n",
      "Requirement already satisfied: mock<3.0.0,>=1.0.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (2.0.0)\n",
      "Requirement already satisfied: fastavro<0.24,>=0.21.4 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (0.23.6)\n",
      "Requirement already satisfied: httplib2<0.18.0,>=0.8 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (0.17.4)\n",
      "Requirement already satisfied: grpcio<2,>=1.29.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (1.32.0)\n",
      "Requirement already satisfied: pydot<2,>=1.2.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (1.4.1)\n",
      "Requirement already satisfied: future<1.0.0,>=0.18.2 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (0.18.2)\n",
      "Requirement already satisfied: dill<0.3.2,>=0.3.1.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (0.3.1.1)\n",
      "Requirement already satisfied: pytz>=2018.3 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (2019.3)\n",
      "Requirement already satisfied: requests<3.0.0,>=2.24.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (2.24.0)\n",
      "Requirement already satisfied: protobuf<4,>=3.12.2 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (3.13.0)\n",
      "Requirement already satisfied: pymongo<4.0.0,>=3.8.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from apache-beam<3,>=2.20) (3.11.0)\n",
      "Requirement already satisfied: tfx-bsl<0.25,>=0.24 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.24.0)\n",
      "Requirement already satisfied: pandas<2,>=1.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (1.1.2)\n",
      "Requirement already satisfied: tensorflow-metadata<0.25,>=0.24 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.24.0)\n",
      "Requirement already satisfied: ipywidgets<8,>=7 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (7.5.1)\n",
      "Requirement already satisfied: jupyter<2,>=1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (1.0.0)\n",
      "Requirement already satisfied: scipy<2,>=1.4.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (1.4.1)\n",
      "Requirement already satisfied: google-api-python-client>=1.7.8 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from witwidget<2,>=1.4.4->fairness-indicators) (1.12.1)\n",
      "Requirement already satisfied: tensorflow-transform<0.25,>=0.24 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow-data-validation<0.25,>=0.24->fairness-indicators) (0.24.0)\n",
      "Requirement already satisfied: joblib<0.15,>=0.12 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow-data-validation<0.25,>=0.24->fairness-indicators) (0.13.2)\n",
      "Requirement already satisfied: keras-preprocessing<1.2,>=1.1.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (1.1.2)\n",
      "Requirement already satisfied: tensorboard<3,>=2.3.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (2.3.0)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (3.1.0)\n",
      "Requirement already satisfied: h5py<2.11.0,>=2.10.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (2.10.0)\n",
      "Requirement already satisfied: gast==0.3.3 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (0.3.3)\n",
      "Requirement already satisfied: wheel>=0.26 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (0.33.6)\n",
      "Requirement already satisfied: wrapt>=1.11.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (1.11.2)\n",
      "Requirement already satisfied: google-pasta>=0.1.8 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (0.1.8)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (1.1.0)\n",
      "Requirement already satisfied: astunparse==1.6.3 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (1.6.3)\n",
      "Requirement already satisfied: tensorflow-estimator<2.4.0,>=2.3.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (2.3.0)\n",
      "Requirement already satisfied: pyasn1>=0.1.7 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from oauth2client<4,>=2.0.1->apache-beam<3,>=2.20) (0.4.8)\n",
      "Requirement already satisfied: pyasn1-modules>=0.0.5 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from oauth2client<4,>=2.0.1->apache-beam<3,>=2.20) (0.2.8)\n",
      "Requirement already satisfied: rsa>=3.1.4 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from oauth2client<4,>=2.0.1->apache-beam<3,>=2.20) (4.6)\n",
      "Requirement already satisfied: docopt in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from hdfs<3.0.0,>=2.1.0->apache-beam<3,>=2.20) (0.6.2)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pbr>=0.11 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from mock<3.0.0,>=1.0.1->apache-beam<3,>=2.20) (5.5.0)\n",
      "Requirement already satisfied: pyparsing>=2.1.4 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from pydot<2,>=1.2.0->apache-beam<3,>=2.20) (2.4.2)\n",
      "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from requests<3.0.0,>=2.24.0->apache-beam<3,>=2.20) (1.24.2)\n",
      "Requirement already satisfied: chardet<4,>=3.0.2 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from requests<3.0.0,>=2.24.0->apache-beam<3,>=2.20) (3.0.4)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from requests<3.0.0,>=2.24.0->apache-beam<3,>=2.20) (2019.9.11)\n",
      "Requirement already satisfied: idna<3,>=2.5 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from requests<3.0.0,>=2.24.0->apache-beam<3,>=2.20) (2.8)\n",
      "Requirement already satisfied: setuptools in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from protobuf<4,>=3.12.2->apache-beam<3,>=2.20) (41.4.0)\n",
      "Requirement already satisfied: tensorflow-serving-api!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tfx-bsl<0.25,>=0.24->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (2.3.0)\n",
      "Requirement already satisfied: googleapis-common-protos<2,>=1.52.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorflow-metadata<0.25,>=0.24->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (1.52.0)\n",
      "Requirement already satisfied: traitlets>=4.3.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (4.3.3)\n",
      "Requirement already satisfied: ipython>=4.0.0; python_version >= \"3.3\" in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (7.8.0)\n",
      "Requirement already satisfied: ipykernel>=4.5.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (5.1.2)\n",
      "Requirement already satisfied: nbformat>=4.2.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (4.4.0)\n",
      "Requirement already satisfied: widgetsnbextension~=3.5.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (3.5.1)\n",
      "Requirement already satisfied: nbconvert in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (5.6.0)\n",
      "Requirement already satisfied: qtconsole in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (4.5.5)\n",
      "Requirement already satisfied: notebook in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (6.0.1)\n",
      "Requirement already satisfied: jupyter-console in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (6.0.0)\n",
      "Requirement already satisfied: google-auth>=1.16.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from google-api-python-client>=1.7.8->witwidget<2,>=1.4.4->fairness-indicators) (1.21.2)\n",
      "Requirement already satisfied: google-auth-httplib2>=0.0.3 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from google-api-python-client>=1.7.8->witwidget<2,>=1.4.4->fairness-indicators) (0.0.4)\n",
      "Requirement already satisfied: uritemplate<4dev,>=3.0.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from google-api-python-client>=1.7.8->witwidget<2,>=1.4.4->fairness-indicators) (3.0.1)\n",
      "Requirement already satisfied: google-api-core<2dev,>=1.21.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from google-api-python-client>=1.7.8->witwidget<2,>=1.4.4->fairness-indicators) (1.22.2)\n",
      "Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorboard<3,>=2.3.0->tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (1.7.0)\n",
      "Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorboard<3,>=2.3.0->tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (0.4.1)\n",
      "Requirement already satisfied: werkzeug>=0.11.15 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorboard<3,>=2.3.0->tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (0.16.0)\n",
      "Requirement already satisfied: markdown>=2.6.8 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from tensorboard<3,>=2.3.0->tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (3.1.1)\n",
      "Requirement already satisfied: decorator in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from traitlets>=4.3.1->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (4.4.0)\n",
      "Requirement already satisfied: ipython-genutils in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from traitlets>=4.3.1->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.2.0)\n",
      "Requirement already satisfied: jedi>=0.10 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.15.1)\n",
      "Requirement already satisfied: pickleshare in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.7.5)\n",
      "Requirement already satisfied: pexpect; sys_platform != \"win32\" in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (4.7.0)\n",
      "Requirement already satisfied: prompt-toolkit<2.1.0,>=2.0.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (2.0.10)\n",
      "Requirement already satisfied: appnope; sys_platform == \"darwin\" in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.1.0)\n",
      "Requirement already satisfied: pygments in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (2.4.2)\n",
      "Requirement already satisfied: backcall in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.1.0)\n",
      "Requirement already satisfied: jupyter-client in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipykernel>=4.5.1->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (5.3.3)\n",
      "Requirement already satisfied: tornado>=4.2 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from ipykernel>=4.5.1->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (6.0.3)\n",
      "Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (3.0.2)\n",
      "Requirement already satisfied: jupyter-core in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (4.5.0)\n",
      "Requirement already satisfied: testpath in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from nbconvert->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.4.2)\n",
      "Requirement already satisfied: mistune<2,>=0.8.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from nbconvert->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.8.4)\n",
      "Requirement already satisfied: bleach in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from nbconvert->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (3.1.0)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: jinja2>=2.4 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from nbconvert->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (2.10.3)\n",
      "Requirement already satisfied: entrypoints>=0.2.2 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from nbconvert->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.3)\n",
      "Requirement already satisfied: defusedxml in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from nbconvert->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.6.0)\n",
      "Requirement already satisfied: pandocfilters>=1.4.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from nbconvert->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (1.4.2)\n",
      "Requirement already satisfied: terminado>=0.8.1 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from notebook->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.8.2)\n",
      "Requirement already satisfied: prometheus-client in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from notebook->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.7.1)\n",
      "Requirement already satisfied: Send2Trash in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from notebook->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (1.5.0)\n",
      "Requirement already satisfied: cachetools<5.0,>=2.0.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from google-auth>=1.16.0->google-api-python-client>=1.7.8->witwidget<2,>=1.4.4->fairness-indicators) (4.1.1)\n",
      "Requirement already satisfied: requests-oauthlib>=0.7.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (1.3.0)\n",
      "Requirement already satisfied: parso>=0.5.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from jedi>=0.10->ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.5.1)\n",
      "Requirement already satisfied: ptyprocess>=0.5 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from pexpect; sys_platform != \"win32\"->ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.6.0)\n",
      "Requirement already satisfied: wcwidth in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from prompt-toolkit<2.1.0,>=2.0.0->ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.1.7)\n",
      "Requirement already satisfied: pyrsistent>=0.14.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.15.4)\n",
      "Requirement already satisfied: attrs>=17.4.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets<8,>=7->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (19.2.0)\n",
      "Requirement already satisfied: webencodings in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from bleach->nbconvert->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (0.5.1)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from jinja2>=2.4->nbconvert->jupyter<2,>=1->tensorflow-model-analysis<0.25,>=0.24->fairness-indicators) (1.1.1)\n",
      "Requirement already satisfied: oauthlib>=3.0.0 in /Users/khushibhansali/opt/anaconda3/lib/python3.7/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow!=2.0.*,!=2.1.*,!=2.2.*,<3,>=1.15.2->fairness-indicators) (3.1.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install git+https://github.com/google-research/tensorflow_constrained_optimization\n",
    "!pip install -q tensorflow-datasets tensorflow\n",
    "!pip install fairness-indicators \\\n",
    "  \"absl-py==0.8.0\" \\\n",
    "  \"pyarrow<0.17,>=0.16\" \\\n",
    "  \"apache-beam<3,>=2.20\" \\\n",
    "  \"avro-python3==1.9.1\" \\\n",
    "  \"pyzmq==17.0.0\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import tempfile\n",
    "import urllib\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "\n",
    "import tensorflow_datasets as tfds\n",
    "tfds.disable_progress_bar()\n",
    "\n",
    "import numpy as np\n",
    "\n",
    "import tensorflow_constrained_optimization as tfco\n",
    "import tensorflow_model_analysis as tfma\n",
    "import fairness_indicators as fi\n",
    "from google.protobuf import text_format\n",
    "import apache_beam as beam"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Eager execution enabled by default.\n",
      "TensorFlow 2.3.0\n"
     ]
    }
   ],
   "source": [
    "if tf.__version__ < \"2.0.0\":\n",
    "  tf.compat.v1.enable_eager_execution()\n",
    "  print(\"Eager execution enabled.\")\n",
    "else:\n",
    "  print(\"Eager execution enabled by default.\")\n",
    "\n",
    "print(\"TensorFlow \" + tf.__version__)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Celeb_A dataset version: 2.0.0\n"
     ]
    }
   ],
   "source": [
    "gcs_base_dir = \"gs://celeb_a_dataset/\"\n",
    "celeb_a_builder = tfds.builder(\"celeb_a\", data_dir=gcs_base_dir, version='2.0.0')\n",
    "\n",
    "celeb_a_builder.download_and_prepare()\n",
    "\n",
    "num_test_shards_dict = {'0.3.0': 4, '2.0.0': 2} # Used because we download the test dataset separately\n",
    "version = str(celeb_a_builder.info.version)\n",
    "print('Celeb_A dataset version: %s' % version)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "local_root = tempfile.mkdtemp(prefix='test-data')\n",
    "def local_test_filename_base():\n",
    "  return local_root\n",
    "\n",
    "def local_test_file_full_prefix():\n",
    "  return os.path.join(local_test_filename_base(), \"celeb_a-test.tfrecord\")\n",
    "\n",
    "def copy_test_files_to_local():\n",
    "  filename_base = local_test_file_full_prefix()\n",
    "  num_test_shards = num_test_shards_dict[version]\n",
    "  for shard in range(num_test_shards):\n",
    "    url = \"https://storage.googleapis.com/celeb_a_dataset/celeb_a/%s/celeb_a-test.tfrecord-0000%s-of-0000%s\" % (version, shard, num_test_shards)\n",
    "    filename = \"%s-0000%s-of-0000%s\" % (filename_base, shard, num_test_shards)\n",
    "    res = urllib.request.urlretrieve(url, filename)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
