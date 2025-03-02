# ollama-qnx

I needed an Ollama client running on QNX, but QNX do not support Go. However QNX 8 comes with Python support by default, I rewrite the Ollama "client" in Python.
```Console
usage: ollama.py [-h] {create,show,run,stop,serve,pull,push,list,ps,cp,rm} ...

Large language model runner

positional arguments:
  {create,show,run,stop,serve,pull,push,list,ps,cp,rm}
    create              Create a model from a Modelfile
    show                Show information for a model
    run                 Run a model
    stop                Stop a running model
    serve               Start ollama
    pull                Pull a model from a registry
    push                Push a model to a registry
    list                List models
    ps                  List running models
    cp                  Copy a model
    rm                  Remove a model

options:
  -h, --help            show this help message and exit
```

## Install
If you want to execute this on QNX, remember to add the "--python=yes" option for mkqnximage, this way python 3.11 package will be include into your QNX image by default. 

Some additional python package is needed for this client. Install them from the requirements.txt. You can also create an python venv first if necessary
```colsole
python -m venv .venv_ollama
source .venv_ollama/bin/activate
pip install -r requirements.txt
```

## Ollama Server setting
By default, the Ollama server only listen on localhost, to allow client from different machine, remember to modify the settings.
```console
export OLLAMA_HOST=0.0.0.0
export OLLAMA_ORIGINS=*
/usr/local/bin/ollama serve
```

## Execute the client
Check the "ollama" shell script, modify the OLLAMA_HOST environment there to point to your sever's ip
```console
$ cat ollama
#!/bin/sh

OLLAMA_HOST=http://192.168.122.1:11434/
export PYTHONUNBUFFERED=1

python3 $(dirname $0)/ollama.py $1 $2 $3 $4 $5 $6 $7
```
Execute the script just as if you are running the ollama client
```console
$ ./ollama list
NAME                             ID            SIZE    MODIFIED
qwen2.5-coder:latest             2b0496514337  4.7 GB  5 weeks ago
deepseek-r1:latest               0a8c26691023  4.7 GB  5 weeks ago
```

```console
$ ./ollama run deepseek-r1
>>> Why is sky blue?
<think>

</think>

The sky appears blue because the Earth's atmosphere scatters sunlight in all directions. Blue light has a
shorter wavelength than other colors of the rainbow, which allows it to be scattered more efficiently by the
gases and particles in the air. This scattering effect is most noticeable during sunrise and sunset when the
light passes through less atmosphere, but it also happens throughout the day.

So, while the Sun emits white light with an almost infinite range of wavelengths, our eyes perceive this
scattered blue light as the dominant color for the sky.
```

## Others
Even though I only tested this on QNX, but this client is implemented in python only, so it *should* also run on any OS that had python support.
