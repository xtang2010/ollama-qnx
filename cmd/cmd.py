import argparse
import os
import sys
import pathlib
#from cryptography.hazmat.primitives.asymmetric import ed25519
#from cryptography.hazmat.primitives import serialization
#from cryptography.hazmat.primitives.asymmetric import ssh
from tqdm import tqdm
from wcwidth import wcwidth
import requests
from tabulate import tabulate
import colored
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time
import signal
import shutil
from typing import List, Dict, Optional
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from api import Client
from format import HumanNumber, HumanBytes, HumanTime
import cmd

class Spinner:
    spinner_symbols = ['-', '\\', '|', '/']

    def __init__(self, desc="Spinner", interval = 0.1, start = True, leave = True):
        self.desc = desc
        self.interval = interval
        self.count = 0
        self.tqdm = tqdm(desc=desc, total = None, bar_format = '{desc}', leave = leave, file = sys.stdout)
        if start:
            self.start()
			
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.start()

    def update(self):
        while self.running:
            time.sleep(self.interval)
            self.tqdm.set_description_str(f"{self.desc}{self.spinner_symbols[self.count % 4]}")
            self.count += 1

    def stop(self):
        self.running = False
        self.thread.join()
        self.tqdm.set_description_str(f"{self.desc}   ")

    def close(self):
        if (self.running):
            self.stop()
        self.tqdm.close()

class Bar:
    def __init__(self, desc, total, completed):
        format = "{l_bar}{bar}|{n_fmt}/{total_fmt} {rate_fmt}{postfix} {remaining}"
        self.tqdm = tqdm(desc = desc, total = total, bar_format = format, unit_scale = 1, unit = "B", file = sys.stdout)
        self.completed = completed

    def update(self, completed):
        if completed > self.completed:
            self.tqdm.update(completed - self.completed)
            self.completed = completed
            self.tqdm.refresh()
            sys.stdout.flush()
   
    def close(self):
        self.tqdm.close()

class ModelFile:
    def create_request(self, dirname):
        # Implement request creation from modelfile
        return {}

class RunOptions:
    def __init__(self, **kwargs):
        self.model        = kwargs.get("model", None)
        self.parent_model = kwargs.get("parent_model", None)
        self.prompt       = kwargs.get("prompt", None) 
        self.messages     = kwargs.get("messages", [])
        self.word_wrap    = kwargs.get("word_wrap", True)
        self.format       = kwargs.get("format", None)
        self.system       = kwargs.get("system", None)
        self.images       = kwargs.get("images", [])
        self.options      = kwargs.get("options", [])
        self.keep_alive   = kwargs.get("keep_alive", None)
        self.multi_modal  = kwargs.get("multi_modal", False)

class DisplayResponseState:
    def __init__(self, **kwargs):
        self.line_length  = kwargs.get("line_length", 0)
        self.word_buffer  = kwargs.get("word_buffer", "")
        os.environ['PYTHONUNBUFFERED'] = '1'

class Cmd:
    def get_modelfile_name(self, args):
        filename = args.file if args.file else "Modelfile"
        abs_name = os.path.abspath(filename)
        if not os.path.exists(abs_name):
            return filename, FileNotFoundError(f"File {abs_name} not found")
        return abs_name, None

    def create_handler(self, args):
        try:
            filename, err = self.get_modelfile_name(args)
            if isinstance(err, FileNotFoundError):
                if not filename:
                    reader = "FROM .\n"
                else:
                    return err
            else:
                with open(filename, 'r') as f:
                    reader = f.read()

            # Parse the modelfile and create request
            modelfile = self.parse_file(reader)
            req = modelfile.create_request(os.path.dirname(filename))
            req['name'] = args.model

            if args.quantize:
                req['quantize'] = args.quantize

            client = Client.from_environment()
            if req.get('files'):
                file_map = {}
                for f, digest in req['files'].items():
                    self.create_blob(args, client, f, digest, p)
                    file_map[os.path.basename(f)] = digest
                req['files'] = file_map

            if req.get('adapters'):
                file_map = {}
                for f, digest in req['adapters'].items():
                    self.create_blob(args, client, f, digest)
                    file_map[os.path.basename(f)] = digest
                req['adapters'] = file_map

            bars = {"order": []} 
            spinner = None
            status = ""

            def progress_callback(resp):
                nonlocal bars, spinner, status
                if resp.get('digest'):
                    if spinner is not None:
                        spinner.close()
                        spinner = None
                    bar = bars.get(resp['digest'])
                    if not bar:
                        bar = Bar(f"pulling {resp['digest'][7:19]}...", resp['total'], resp['completed'])
                        bars[resp['digest']] = bar
                        bars['order'].append(bar)
                    if 'completed' in resp:
                        bar.update(resp['completed'])
                elif status != resp.get('status'):
                    if spinner is not None:
                        spinner.close()
                    status = resp["status"]
                    spinner = Spinner(f"{status}: ")
                    bars['order'].append(spinner)

            client.create(req, progress_callback)
        except Exception as e:
            return e

        for d in bars['order']:
            d.close()

    def create_blob(self, args, client, path, digest):
        real_path = os.path.realpath(path)
        with open(real_path, 'rb') as bin:
            file_info = os.stat(real_path)
            file_size = file_info.st_size

            status = f"copying file {digest}"
            spinner = Spinner(f"{status}: ")

            client.create_blob(digest, bin.read())
            spinner.close()

    def parse_file(self, content):
        # Implement modelfile parsing
        return ModelFile()

    def load_or_unload_model(self, opts):
        spinner = Spinner(desc="", leave = False)

        client = Client.from_environment()
        if not client:
            return ValueError("Failed to create client")

        req = {
            "model": opts.model,
            "keep_alive": opts.keep_alive.total_seconds() if opts.keep_alive else None
        }

        def callback(response):
            pass  # No-op callback

        try:
            response = client.generate(req, callback)
    #        if response.status_code >= 400:
    #            return ValueError(f"Failed to generate: {response.text}")
        except Exception as e:
            spinner.close()
            return e

        spinner.close()
        return None

    def stop_handler(self, args):
        opts = RunOptions(model=args.model, keep_alive=timedelta(seconds = 0))
        error = self.load_or_unload_model(opts)
        if error:
            if "not found" in str(error):
                sys.exit(f"couldn't find model \"{args.model}\" to stop")
            else:
                sys.exit(f"Error: {error}")
        return None

    def run_handler(self, args):
        interactive = True

        #word_wrap = os.environ.get("TERM") == "xterm-256color"
        word_wrap = True
        
        opts = RunOptions( 
            model = args.model,
            word_wrap = word_wrap,
            format= args.format,
            keep_alive=None,
            prompt="",
            multi_modal=False,
            parent_model="",
            options={}        
        )

        if "keepalive" in args:
            try:
                opts.keep_alive = timedelta(seconds = int(args.keepalive))
            except ValueError:
                sys.exit("Invalid duration for --keepalive")

        prompts = args.prompt
        if not sys.stdin.isatty():
            try:
                stdin_input = sys.stdin.read()
                prompts = [stdin_input] + prompts
                opts.word_wrap = False
                interactive = False
            except Exception as e:
                sys.exit(f"Error reading from stdin: {e}")

        opts.prompt = " ".join(prompts)
        if prompts:
            interactive = False

        if not sys.stdout.isatty():
            interactive = False

        if args.nowordwrap:
            opts.word_wrap = False

        client = Client.from_environment()
        if not client:
            sys.exit("Failed to create client")

        try:
            info = client.show({"model": args.model})
        except Exception as e:
            sys.exit(f"Error fetching model info: {e}")

        opts.multi_modal = len(info.get("projector_info", [])) != 0
        opts.parent_model = info.get("details", {}).get("parent_model", "")

        if interactive:
            try:
                self.load_or_unload_model(opts)
            except Exception as e:
                sys.exit(f"Error loading/unloading model: {e}")

            for msg in info.get("messages", []):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    print(colored.fg("blue") + f">>> {content}")
                elif role == "assistant":
                    self.display_response(content, opts.word_wrap)

            gint = cmd.GenerateInteractive(self, args, opts.__dict__)
            return gint.main_loop()
        else:
            return self.generate(args, opts)

    def push_handler(self, args):
        client = Client.from_environment()
        if not client:
            sys.exit("Failed to create client")

        insecure = args.insecure

        bars = {"order": []} 
        status = ""
        spinner = None

        def progress_callback(resp):
            nonlocal status, spinner, bars
            if resp.get("digest"):
                if spinner is not None:
                    spinner.stop()
                    spinner = None
                bar = bars.get(resp["digest"])
                if not bar:
                    bar = Bar(f"pushing {resp['digest'][7:19]}", resp["total"], 0)
                    bars[resp["digest"]] = bar
                    bars['order'].append(bar)
                if "completed" in resp:
                    bar.update(resp["completed"])
            elif status != resp.get("status"):
                if spinner is not None:
                    spinner.stop()
                status = resp["status"]
                spinner = Spinner(f"{status}: ")
                bars['order'].append(spinner)

        request = {
            "name": args.model,
            "insecure": insecure
        }

        try:
            response = client.push(request, progress_callback)
            #if response.status_code >= 400:
            #    if "access denied" in response.text:
            #        sys.exit("You are not authorized to push to this namespace, create the model under a namespace you own")
            #    sys.exit(f"Failed to push model: {response.text}")
        except Exception as e:
            sys.exit(f"Error: {e}")

        #p.close()
        for d in bars['order']:
            d.close()

        destination = args.model
        if destination.endswith(".ollama.ai") or destination.endswith(".ollama.com"):
            destination = "https://ollama.com/" + destination.split(":")[0]
        print("\nYou can find your model at:\n")
        print(f"\t{destination}")

    def list_handler(self, args):
        client = Client.from_environment()
        if not client:
            print("Failed to create client")
            return

        models = client.list()

        data = []
        for m in models["models"]:
            if args.model is None or m["name"].lower().startswith(args.model.lower()):
                data.append([m["name"], m["digest"][:12], HumanBytes(m["size"]), HumanTime(m["modified_at"], "Never")])

        print(tabulate(data, headers=["NAME", "ID", "SIZE", "MODIFIED"], tablefmt="plain"))

    def list_running_handler(self, args):
        client = Client.from_environment()
        if not client:
            print("Failed to create client")
            return

        models = client.list_running()
        
        data = []
        for m in models["models"]:
            if (args.model is None or m["name"].startswith(args.model)):
                proc_str = ""
                if m["size_vram"] == 0:
                    proc_str = "100% CPU"
                elif m["size_vram"] == m["size"]:
                    proc_str = "100% GPU"
                elif m["size_vram"] > m["size"] or m["size"] == 0:
                    proc_str = "Unknown"
                else:
                    size_cpu = m["size"] - m["size_vram"]
                    cpu_percent = round((size_cpu / m["size"]) * 100)
                    proc_str = f"{cpu_percent}%/{100 - cpu_percent}% CPU/GPU"

                until = "Never"
                if datetime.fromisoformat(m["expires_at"]) < datetime.now(ZoneInfo('localtime')):
                    until = "Stopping..."
                else:
                    until = HumanTime(m["expires_at"], "Never")

                data.append([m["name"], m["digest"][:12], HumanBytes(m["size"]), proc_str, until])

        print(tabulate(data, headers=["NAME", "ID", "SIZE", "PROCESSOR", "UNTIL"], tablefmt="plain"))

    def delete_handler(self, args):
        client = Client.from_environment()
        if not client:
            print("Failed to create client")
            return

        # Unload the model if it's running before deletion
        opts = RunOptions(model = args.model[0], keep_alive = timedelta(seconds = 0))
        try:
            self.load_or_unload_model(opts)
        except Exception as e:
            if "not found" not in str(e):
                print(f"Unable to stop existing running model \"{args.model_name}\": {e}")
                return

        for name in args.model:
            request = {
                "name": name
            }
            try:
                client.delete(request)
                print(f"deleted '{name}'")
            except Exception as e:
                print(f"Error deleting model '{name}': {e}")
                return

    def show_handler(self, args):
        client = Client.from_environment()
        if not client:
            print("Failed to create client")
            return

        license = args.license
        modelfile = args.modelfile
        parameters = args.parameters
        system = args.system
        template = args.template

        flags_set = sum([license, modelfile, parameters, system, template])
        if flags_set > 1:
            print("Only one of '--license', '--modelfile', '--parameters', '--system', or '--template' can be specified")
            return

        show_type = ""
        if license:
            show_type = "license"
        elif modelfile:
            show_type = "modelfile"
        elif parameters:
            show_type = "parameters"
        elif system:
            show_type = "system"
        elif template:
            show_type = "template"

        resp = client.show({"model": args.model})

        if flags_set == 1:
            if show_type == "license":
                print(resp.get("license", ""))
            elif show_type == "modelfile":
                print(resp.get("modelfile", ""))
            elif show_type == "parameters":
                print(resp.get("parameters", ""))
            elif show_type == "system":
                print(resp.get("system", ""))
            elif show_type == "template":
                print(resp.get("template", ""))
            return

        self.show_info(resp)

    def show_info(self, resp):
        def table_render(header, rows):
            print(f" {header}")
            print(tabulate(rows, tablefmt="plain"))
            print()

        model_info = resp.get("model_info", {})
        projector_info = resp.get("projector_info", {})

        if model_info != {}:
            arch = model_info.get("general.architecture", "")
            model_rows = [
                ["", "architecture", arch],
                ["", "parameters", HumanNumber(model_info.get("general.parameter_count", ""))],
                ["", "context length", model_info.get(f"{arch}.context_length", "")],
                ["", "embedding length", model_info.get(f"{arch}.embedding_length", "")],
                ["", "quantization", resp.get("details", {}).get("quantization_level", "")]
            ]
        else:
            model_rows = [
                ["", "architecture", resp['details']['family']],
                ["", "parameters", resp['details']['parameter_size']]        
            ]
        table_render("Model", model_rows)

        if projector_info:
            projector_rows = [
                ["", "architecture", projector_info.get("general.architecture", "")],
                ["", "parameters", projector_info.get("general.parameter_count", "")],
                ["", "embedding length", projector_info.get("embedding_length", "")],
                ["", "dimensions", projector_info.get("projection_dim", "")]
            ]
            table_render("Projector", projector_rows)

        if resp.get("parameters"):
            parameters_lines = resp["parameters"].splitlines()
            parameters_rows = [["", *line.split()] for line in resp["parameters"].splitlines()]
            table_render("Parameters", parameters_rows)

        if resp.get("system"):
            system_lines = [l for l in resp["system"].splitlines() if l]
            system_rows = [["", line] for line in system_lines[:2]]
            table_render("System", system_rows)

        if resp.get("license"):
            license_lines = [l for l in resp["license"].splitlines() if l]
            license_rows = [["", line] for line in license_lines[:2]]
            table_render("License", license_rows)

    def copy_handler(self, args):
        client = Client.from_environment()
        if not client:
            print("Failed to create client")
            return

        request = {
            "source": args.source,
            "destination": args.destination
        }
        try:
            response = client.copy(request)
            #if response.status_code >= 400:
            #    raise ValueError(f"Failed to copy model: {response.text}")
            print(f"copied '{args.source}' to '{args.destination}'")
        except Exception as e:
            print(f"Error: {e}")

    def pull_handler(self, args):
        client = Client.from_environment()
        if not client:
            print("Failed to create client")
            return

        insecure = args.insecure
        
        bars = {"order": []} 
        status = ""
        spinner = None

        def progress_callback(resp):
            nonlocal status, spinner, bars

            if resp.get("digest"):
                if spinner is not None:
                    spinner.stop()
                    spinner = None
                bar = bars.get(resp["digest"])
                if bar is None:
                    bar = Bar(f"pulling {resp['digest'][7:19]}", resp["total"], 0)
                    bars[resp["digest"]] = bar
                    bars['order'].append(bar)
                if "completed" in resp:
                    bar.update(resp["completed"])
            else:
                if status != resp.get("status"):
                    if spinner is not None:
                        spinner.stop()
                    status = resp["status"]
                    spinner = Spinner(f"{status}: ")
                    bars['order'].append(spinner)

        request = {
            "name": args.model,
            "insecure": insecure
        }

        try:
            response = client.pull(request, progress_callback)
            #if response.status_code >= 400:
            #    print(f"Failed to pull model: {response.text}")
            #    return
        except Exception as e:
            print(f"Error: {e}")
            return

        #p.close()
        #if spinner is not None:
        #    spinner.close()
        for d in bars['order']:
            d.close()

    def display_response(self, content, word_wrap, state):
        term_width = shutil.get_terminal_size().columns
        ch_width = 0
        if word_wrap and term_width >= 10:
            for ch in content:
                if state.line_length + 1 > term_width - 5:
                    if len(state.word_buffer) > term_width - 10:
                        sys.stdout.write(f"{state.word_buffer}{ch}")
                        state.word_buffer = ""
                        state.line_length = 0
                        continue

                    # backtrack the length of the last word and clear to the end of the line
                    a = len(state.word_buffer)
                    if a > 0:
                        sys.stdout.write(f"\x1b[{a}D")
                    sys.stdout.write("\x1b[K\n")
                    sys.stdout.write(f"{state.word_buffer}{ch}")
                    ch_width = wcwidth(ch)  # Simplified for Python

                    state.line_length = len(state.word_buffer) + ch_width
                else:
                    sys.stdout.write(ch)
                    ch_width = wcwidth(ch)
                    state.line_length += ch_width
                    if ch_width >= 2:
                        state.word_buffer = ""
                        continue

                    if ch == ' ':
                        state.word_buffer = ""
                    elif ch == '\n':
                        state.line_length = 0
                    else:
                        state.word_buffer += ch
        else:
            sys.stdout.write(f"{state.word_buffer}{content}")
            if state.word_buffer:
                state.word_buffer = ""

    def chat(self, args, opts):
        client = Client.from_environment()
        if not client:
            print("Failed to create client")
            return None, "Failed to create client"

        spinner = Spinner(desc="", leave = False)

        cancel_event = threading.Event()

        def signal_handler(sig, frame):
            spinner.close()
            cancel_event.set()

        signal.signal(signal.SIGINT, signal_handler)

        state = DisplayResponseState()
        latest = None
        full_response = []
        role = ""

        def progress_callback(response):
            nonlocal latest, spinner
            latest = response

            if spinner is not None:
                spinner.close()
                spinner = None
            role = response["message"]["role"]
            content = response["message"]["content"]
            full_response.append(content)

            self.display_response(content, opts.word_wrap, state)

        request = {
            "model": opts.model,
            "messages": opts.messages,
            "format": opts.format,
            "options": opts.options
        }

        if opts.keep_alive:
            request["keep_alive"] = opts.keep_alive.total_seconds()

        try:
            response = client.chat(request, progress_callback)
            #if response.status_code >= 400:
            #    raise ValueError(f"Failed to chat: {response.text}")
        except Exception as e:
            if cancel_event.is_set():
                if spinner is not None:
                    spinner.close()
                return None, None
            return None, str(e)

        if opts.messages:
            print()
            print()

        verbose = args.verbose
        if verbose:
            if latest:
                print(latest.get("summary", ""))

        return {"role": role, "content": "".join(full_response)}, None

    def generate(self, cmd, opts):
        client = Client.from_environment()
        if not client:
            print("Failed to create client")
            return "Failed to create client"

        spinner = Spinner(desc="", leave = False)

        cancel_event = threading.Event()

        def signal_handler(sig, frame):
            if spinner is not None:
                spinner.close()
                spinner = None
            cancel_event.set()

        signal.signal(signal.SIGINT, signal_handler)

        state = DisplayResponseState()
        latest = None

        def progress_callback(response):
            nonlocal latest, spinner
            latest = response
            if spinner is not None:
                spinner.close()
                spinner = None

            content = response.get("response", "")
            self.display_response(content, opts.word_wrap, state)

        request = {
            "model": opts.model,
            "prompt": opts.prompt,
            "images": opts.images,
            "format": opts.format,
            "system": opts.system,
            "options": opts.options,
            "keep_alive": opts.keep_alive.total_seconds()
        }

        try:
            response = client.generate(request, progress_callback)
            #if response.status_code >= 400:
            #    raise ValueError(f"Failed to generate: {response.text}")
        except Exception as e:
            if cancel_event.is_set():
                return None
            return str(e)

        if opts.prompt:
            print()
            print()

        if latest and not latest.get("done", True):
            return None

        verbose = cmd.verbose
        if verbose:
            if latest:
                print(latest.get("summary", ""))

        return None

    def run_server(self):
        if initialize_keypair():
            return "Failed to initialize key pair"

        host = os.getenv("OLLAMA_HOST", "localhost:11434")
        host, port = host.split(":")
        port = int(port)

        server_address = (host, port)
        httpd = ThreadedHTTPServer(server_address, SimpleHTTPRequestHandler)

        def server_thread():
            httpd.serve_forever()

        thread = threading.Thread(target=server_thread)
        thread.daemon = True
        thread.start()

        print(f"Server started at http://{host}:{port}")

        def signal_handler(sig, frame):
            httpd.shutdown()
            httpd.server_close()
            print("Server stopped")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        thread.join()

    def NewCLI(self):
        parser = argparse.ArgumentParser(description="Large language model runner")
        subparsers = parser.add_subparsers(dest="command")

        create_parser = subparsers.add_parser("create", help="Create a model from a Modelfile")
        create_parser.add_argument("model", help="Model name")
        create_parser.add_argument("-f", "--file", help="Name of the Modelfile (default 'Modelfile')")
        create_parser.add_argument("-q", "--quantize", help="Quantize model to this level (e.g. q4_0)")

        show_parser = subparsers.add_parser("show", help="Show information for a model")
        show_parser.add_argument("model", help="Model name")
        show_parser.add_argument("--license", action="store_true", help="Show license of a model")
        show_parser.add_argument("--modelfile", action="store_true", help="Show Modelfile of a model")
        show_parser.add_argument("--parameters", action="store_true", help="Show parameters of a model")
        show_parser.add_argument("--template", action="store_true", help="Show template of a model")
        show_parser.add_argument("--system", action="store_true", help="Show system message of a model")

        run_parser = subparsers.add_parser("run", help="Run a model")
        run_parser.add_argument("model", help="Model name")
        run_parser.add_argument("prompt", nargs="*", help="Prompt for the model")
        run_parser.add_argument("--keepalive", default=0, help="Duration to keep a model loaded (e.g. 5m)")
        run_parser.add_argument("--verbose", action="store_true", help="Show timings for response")
        run_parser.add_argument("--insecure", action="store_true", help="Use an insecure registry")
        run_parser.add_argument("--nowordwrap", action="store_true", help="Don't wrap words to the next line automatically")
        run_parser.add_argument("--format", default="", help="Response format (e.g. json)")

        stop_parser = subparsers.add_parser("stop", help="Stop a running model")
        stop_parser.add_argument("model", help="Model name")

        serve_parser = subparsers.add_parser("serve", help="Start ollama")
        serve_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        serve_parser.add_argument("--host", help="Host to bind to")
        serve_parser.add_argument("--keep-alive", help="Keep alive duration")
        serve_parser.add_argument("--max-loaded-models", type=int, help="Maximum number of loaded models")
        serve_parser.add_argument("--max-queue", type=int, help="Maximum queue size")
        serve_parser.add_argument("--models", help="Path to models directory")
        serve_parser.add_argument("--num-parallel", type=int, help="Number of parallel processes")
        serve_parser.add_argument("--no-prune", action="store_true", help="Do not prune models")
        serve_parser.add_argument("--origins", help="Allowed origins")
        serve_parser.add_argument("--sched-spread", action="store_true", help="Enable schedule spreading")
        serve_parser.add_argument("--tmpdir", help="Temporary directory")
        serve_parser.add_argument("--flash-attention", action="store_true", help="Enable flash attention")
        serve_parser.add_argument("--kv-cache-type", help="KV cache type")
        serve_parser.add_argument("--llm-library", help="LLM library")
        serve_parser.add_argument("--gpu-overhead", type=float, help="GPU overhead")
        serve_parser.add_argument("--load-timeout", type=int, help="Load timeout")

        pull_parser = subparsers.add_parser("pull", help="Pull a model from a registry")
        pull_parser.add_argument("model", help="Model name")
        pull_parser.add_argument("--insecure", action="store_true", help="Use an insecure registry")

        push_parser = subparsers.add_parser("push", help="Push a model to a registry")
        push_parser.add_argument("model", help="Model name")
        push_parser.add_argument("--insecure", action="store_true", help="Use an insecure registry")

        list_parser = subparsers.add_parser("list", help="List models")
        list_parser.add_argument("model", nargs="?", help="Model name")

        ps_parser = subparsers.add_parser("ps", help="List running models")
        ps_parser.add_argument("model", nargs="?", help="Model name")

        copy_parser = subparsers.add_parser("cp", help="Copy a model")
        copy_parser.add_argument("source", help="Source model")
        copy_parser.add_argument("destination", help="Destination model")

        delete_parser = subparsers.add_parser("rm", help="Remove a model")
        delete_parser.add_argument("model", nargs="+", help="Model name(s)")

        args = parser.parse_args()

        if args.command == "create":
            self.create_handler(args)
        elif args.command == "show":
            self.show_handler(args)
        elif args.command == "run":
            self.run_handler(args)
        elif args.command == "stop":
            self.stop_handler(args)
        elif args.command == "serve":
            self.run_server(args)
        elif args.command == "pull":
            self.pull_handler(args)
        elif args.command == "push":
            self.push_handler(args)
        elif args.command == "list":
            self.list_handler(args)
        elif args.command == "ps":
            self.list_running_handler(args)
        elif args.command == "cp":
            self.copy_handler(args)
        elif args.command == "rm":
            self.delete_handler(args)
        else:
            parser.print_help()

if __name__ == "__main__":
    Cmd().NewCLI()
