import os
import re
import http.client
import mimetypes
import io
import threading
import json
import requests
import types
from typing import List, Optional
from datetime import timedelta
import cmd

class MultilineState:
    NONE = 0
    PROMPT = 1
    SYSTEM = 2

class Scanner:
    def __init__(self, prompt, alt_prompt, placeholder, alt_placeholder):
        self.prompt = prompt
        self.alt_prompt = alt_prompt
        self.placeholder = placeholder
        self.alt_placeholder = alt_placeholder
        self.history = []
        self.history_enabled = True
        self.pasting = False

    def readline(self):
        try:
            line = input(self.prompt)
            if (self.history_enabled):
                self.history.append(line)
            return line
        except EOFError:
            return None
        except KeyboardInterrupt:
            return ""

    def history_enable(self):
        self.history_enabled = True

    def history_disable(self):
        self.history_enabled = False

class GenerateInteractive:
    def __init__(self, cmd, args, opts):
        self.opts = opts
        self.args = args
        self.cmd  = cmd

    def usage(self):
        print("Available Commands:")
        print("  /set            Set session variables")
        print("  /show           Show model information")
        print("  /load <model>   Load a session or model")
        print("  /save <model>   Save your current session")
        print("  /clear          Clear session context")
        print("  /bye            Exit")
        print("  /?, /help       Help for a command")
        print("  /? shortcuts    Help for keyboard shortcuts")
        print()
        print('Use """ to end multi-line input')
        print()
        if self.opts.get('MultiModal', False):
            print("Use /path/to/file to include .jpg or .png images.")
        print()

    def usage_set(self):
        print("Available Commands:")
        print("  /set parameter ...     Set a parameter")
        print("  /set system <string>   Set system message")
        print("  /set history           Enable history")
        print("  /set nohistory         Disable history")
        print("  /set wordwrap          Enable wordwrap")
        print("  /set nowordwrap        Disable wordwrap")
        print("  /set format json       Enable JSON mode")
        print("  /set noformat          Disable formatting")
        print("  /set verbose           Show LLM stats")
        print("  /set quiet             Disable LLM stats")
        print()

    def usage_shortcuts(self):
        print("Available keyboard shortcuts:")
        print("  Ctrl + a            Move to the beginning of the line (Home)")
        print("  Ctrl + e            Move to the end of the line (End)")
        print("  Alt + b            Move back (left) one word")
        print("  Alt + f            Move forward (right) one word")
        print("  Ctrl + k            Delete the sentence after the cursor")
        print("  Ctrl + u            Delete the sentence before the cursor")
        print("  Ctrl + w            Delete the word before the cursor")
        print()
        print("  Ctrl + l            Clear the screen")
        print("  Ctrl + c            Stop the model from responding")
        print("  Ctrl + d            Exit ollama (/bye)")
        print()

    def usage_show(self):
        print("Available Commands:")
        print("  /show info         Show details for this model")
        print("  /show license      Show model license")
        print("  /show modelfile    Show Modelfile for this model")
        print("  /show parameters   Show parameters for this model")
        print("  /show system       Show system message")
        print("  /show template     Show prompt template")
        print()

    def usage_parameters(self):
        print("Available Parameters:")
        print("  /set parameter seed <int>             Random number seed")
        print("  /set parameter num_predict <int>      Max number of tokens to predict")
        print("  /set parameter top_k <int>            Pick from top k num of tokens")
        print("  /set parameter top_p <float>          Pick token based on sum of probabilities")
        print("  /set parameter min_p <float>          Pick token based on top token probability * min_p")
        print("  /set parameter num_ctx <int>          Set the context size")
        print("  /set parameter temperature <float>    Set creativity level")
        print("  /set parameter repeat_penalty <float> How strongly to penalize repetitions")
        print("  /set parameter repeat_last_n <int>    Set how far back to look for repetitions")
        print("  /set parameter num_gpu <int>          The number of layers to send to the GPU")
        print("  /set parameter stop <string> <string> ...   Set the stop parameters")
        print()

    def normalize_file_path(self, fp):
        return fp.replace("\\ ", " ").replace("\\(", "(").replace("\\)", ")").replace("\\[", "[").replace("\\]", "]").replace("\\{", "{").replace("\\}", "}").replace("\\$", "$").replace("\\&", "&").replace("\\;", ";").replace("\\'", "'").replace("\\\\", "\\").replace("\\*", "*").replace("\\?", "?")

    def extract_file_names(self, input_str):
        regex_pattern = r'(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png)\b'
        return re.findall(regex_pattern, input_str)

    def extract_file_data(self, input_str):
        file_paths = self.extract_file_names(input_str)
        imgs = []

        for fp in file_paths:
            nfp = self.normalize_file_path(fp)
            data, err = self.get_image_data(nfp)
            if err and err == FileNotFoundError:
                continue
            elif err:
                print(f"Couldn't process image: {err}")
                return "", imgs, err
            print(f"Added image '{nfp}'")
            input_str = input_str.replace(fp, "")
            imgs.append(data)
        return input_str.strip(), imgs, None

    def get_image_data(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                data = file.read()
                content_type = mimetypes.guess_type(file_path)[0]
                if content_type not in ["image/jpeg", "image/jpg", "image/png"]:
                    return None, ValueError("Invalid image type")
                if len(data) > 100 * 1024 * 1024:  # 100MB
                    return None, ValueError("File size exceeds maximum limit (100MB)")
                return data, None
        except FileNotFoundError:
            return None, FileNotFoundError()
        except Exception as e:
            return None, e

    def chat1(cmd, opts):
        # Placeholder for chat functionality
        return None, None

    def load_or_unload_model1(cmd, opts):
        # Placeholder for load or unload model functionality
        pass

    def new_create_request(self, name, opts):
        req = {
            'name': name,
            'from': opts.get('ParentModel', opts.get('Model')),
            'system': opts.get('System', ''),
            'parameters': opts.get('Options', {}),
            'messages': opts.get('Messages', [])
        }
        return req

    def show_info1(resp, stream):
        # Placeholder for show info functionality
        pass

    def main_loop(self):
        scanner = Scanner(prompt=">>> ", alt_prompt="... ", placeholder="Send a message (/? for help)", alt_placeholder='Use """ to end multi-line input')
        multiline = MultilineState.NONE
        sb = io.StringIO()

        while True:
            line = scanner.readline()
            if line is None:
                print()
                break
            elif line == "":
                continue

            if multiline != MultilineState.NONE:
                before, ok = line.rsplit('"""', 1) if '"""' in line else (line, False)
                sb.write(before)
                if not ok:
                    print(sb.getvalue(), end="")
                    continue

                if multiline == MultilineState.SYSTEM:
                    opts['System'] = sb.getvalue()
                    opts['Messages'].append({'role': 'system', 'content': opts['System']})
                    print("Set system message.")
                    sb = io.StringIO()
                multiline = MultilineState.NONE
                scanner.prompt = ">>> "
            elif line.startswith('"""'):
                line = line[3:]
                line, ok = line.rsplit('"""', 1) if '"""' in line else (line, False)
                sb.write(line)
                if not ok:
                    multiline = MultilineState.PROMPT
                    scanner.prompt = "... "
                    continue
            elif scanner.pasting:
                sb.write(line + "\n")
                continue
            elif line.startswith("/list"):
                args = line.split()
                if len(args) > 1:
                    m = {"model": args[1]}
                else:
                    m = {"model": None}
                self.cmd.list_handler(types.SimpleNamespace(**m))
            elif line.startswith("/load"):
                args = line.split()
                if len(args) != 2:
                    print("Usage:\n  /load <modelname>")
                    continue
                opts = {}
                opts['model'] = args[1]
                opts['keep_alive'] = timedelta(seconds = 0)
                print(f'Loading model "{opts["""model"""]}"')
                error = self.cmd.load_or_unload_model(types.SimpleNamespace(**opts))
                if error:
                    if "not found" in str(error):
                        print(f"couldn't find model \"{opts['model']}\" to load")
                    else:
                        print(f"Error: {error}")
                    #break
                else:
                    self.opts["model"] = opts["model"]
            elif line.startswith("/save"):
                args = line.split()
                if len(args) != 2:
                    print("Usage:\n  /save <modelname>")
                    continue
                client = requests.Session()
                req = self.new_create_request(args[1], opts)
                fn = lambda resp: None
                try:
                    client.post("http://localhost:11434/create", json=req, stream=True)
                    print(f"Created new model '{args[1]}'")
                except Exception as e:
                    print(f"Error: {e}")
            elif line.startswith("/clear"):
                self.opts['messages'] = []
                if self.opts['system']:
                    self.opts['messages'].append({'role': 'system', 'content': self.opts['system']})
                print("Cleared session context")
            elif line.startswith("/set"):
                args = line.split()
                if len(args) > 1:
                    if args[1] == "history":
                        scanner.history_enable()
                    elif args[1] == "nohistory":
                        scanner.history_disable()
                    elif args[1] == "wordwrap":
                        self.opts['word_wrap'] = True
                        print("Set 'wordwrap' mode.")
                    elif args[1] == "nowordwrap":
                        self.opts['word_wrap'] = False
                        print("Set 'nowordwrap' mode.")
                    elif args[1] == "verbose":
                        self.opts['verbose'] = True
                        print("Set 'verbose' mode.")
                    elif args[1] == "quiet":
                        self.opts['verbose'] = False
                        print("Set 'quiet' mode.")
                    elif args[1] == "format":
                        if len(args) < 3 or args[2] != "json":
                            print("Invalid or missing format. For 'json' mode use '/set format json'")
                        else:
                            self.opts['format'] = args[2]
                            print(f"Set format to '{args[2]}' mode.")
                    elif args[1] == "noformat":
                        self.opts['format'] = ""
                        print("Disabled format.")
                    elif args[1] == "parameter":
                        if len(args) < 4:
                            self.usage_parameters()
                            continue
                        params = args[3:]
                        self.opts['options'][args[2]] = params
                        print(f"Set parameter '{args[2]}' to '{', '.join(params)}'")
                    elif args[1] == "system":
                        if len(args) < 3:
                            self.usage_set()
                            continue
                        multiline = MultilineState.SYSTEM
                        line = " ".join(args[2:])
                        line, ok = line.lstrip('"""').rstrip('"""') if line.startswith('"""') and line.endswith('"""') else (line, False)
                        sb.write(line)
                        if not ok:
                            multiline = MultilineState.NONE
                        if multiline == MultilineState.NONE:
                            self.opts['system'] = sb.getvalue()
                            self.opts['messages'].append({'role': 'system', 'content': sb.getvalue()})
                            print("Set system message.")
                            sb = io.StringIO()
                        continue
                    else:
                        print(f"Unknown command '/set {args[1]}'. Type /? for help")
                else:
                    self.usage_set()
            elif line.startswith("/show"):
                args = line.split()
                if len(args) > 1:
                    if (args[1].lower() not in ["info", "license", "modelfile", "parameters", "system", "template"]):
                        print(f"Unknown command '/show {args[1]}'. Type /? for help")
                    else:
                        o = {
                            "model": self.opts['model'],
                            "license": False,
                            "modelfile": False,
                            "parameters": False,
                            "system": False,
                            "template": False
                        }
                        o[args[1].lower()] = True
                        self.cmd.show_handler(types.SimpleNamespace(**o))
                else:
                    self.usage_show()
            elif line.startswith("/help") or line.startswith("/?"):
                args = line.split()
                if len(args) > 1:
                    if args[1] in ["set", "/set"]:
                        self.usage_set()
                    elif args[1] in ["show", "/show"]:
                        self.usage_show()
                    elif args[1] in ["shortcut", "shortcuts"]:
                        self.usage_shortcuts()
                else:
                    self.usage()
            elif line.startswith("/exit") or line.startswith("/bye"):
                break
            elif line.startswith("/"):
                print(f"Unknown command '{line}'. Type /? for help")
            else:
                sb.write(line + "\n")

            if sb.getvalue() and multiline == MultilineState.NONE:
                new_message = {'role': 'user', 'content': sb.getvalue()}
                if self.opts.get('multi_model'):
                    msg, images, err = self.extract_file_data(sb.getvalue())
                    if err:
                        return err
                    new_message['content'] = msg
                    new_message['images'] = images
                self.opts['messages'].append(new_message)
                assistant, err = self.cmd.chat(self.args, types.SimpleNamespace(**self.opts))
                if err:
                    return err
                if assistant:
                    self.opts['messages'].append(assistant)
                sb = io.StringIO()

# Example usage
if __name__ == "__main__":
    command = None  # Placeholder for command context
    opts = {
        'model': 'deepseek-r1',
        'messages': [],
        'system': '',
        'keep_alive': None,
        'options': {},
        'verbose': False,
        'format': '',
        'word_wrap': True,
        'multi_modal': False
    }
    gint = GenerateInteractive(cmd.Cmd(), command, opts)
    gint.main_loop()