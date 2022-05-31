# -*- coding:utf-8 -*-

from subprocess import Popen, PIPE
import time
import sys
import io
import os
import threading
import configparser
from dataclasses import dataclass


@dataclass
class SubprocessStatus:
    wait_for_input_to_subprocess: bool

    @property
    def is_wait(self):
        pass

    @is_wait.getter
    def is_wait(self):
        return self.wait_for_input_to_subprocess

    @is_wait.setter
    def is_wait(self, is_wait):
        self.wait_for_input_to_subprocess = is_wait


class Threadable:
    def thread_target(self):
        pass

    def start(self):
        self.t = threading.Thread(target=self.thread_target)
        self.t.setDaemon(True)
        self.t.start()


class ProcessWatcher(Threadable):
    """
        processがwait for inputになったらcallback
    """
    subprocess_status: SubprocessStatus

    def __init__(self, subprocess_status, callback):
        self.subprocess_status = subprocess_status
        self.callback = callback

    def wait(self):
        time.sleep(0.1)

    def thread_target(self):
        while True:
            self.wait()
            if not self.subprocess_status.is_wait:
                continue
            self.callback()


class StdoutLineReader(Threadable):
    """
        stdoutを1 byteずつ読み込んで、行として処理
    """
    line = b""
    line_tmp = b""
    h_system_utterance = "system utterance :"
    h_end = "End of dialogue"

    def __init__(self, fd_stdout: int, callback_loop, callback_sys, callback_end):
        self.fd_stdout = fd_stdout
        self.callback_loop = callback_loop
        self.callback_sys = callback_sys
        self.callback_end = callback_end

    def thread_target(self):
        
        self.stdout = os.fdopen(self.fd_stdout, "rb")
        linesep = os.linesep.encode()
        #print(self.line)
        while True:
            r = self.stdout.read(1)
            self.line += r
            self.callback_loop(self.line)
            #print(self.line)
            if self.line.endswith(linesep):
                if self.line.decode('cp932').startswith(self.h_system_utterance):
                    self.callback_sys(self.line.decode('cp932')[19:-2])
                elif self.line.decode('cp932').startswith(self.h_end):
                    self.callback_end()
                #print(self.line)
                self.line = b""
                

class MainProcess:
    h_topic_select = b">> "
    h_utt_input = b"your utterance >> "
    h_ui_input = b"your impression >> "
    h_confident_noun_input = b"confident_noun >> "


    def __init__(self):
        self.fd_stdin_r, self.fd_stdin_w = os.pipe()
        self.fd_stdout_r, self.fd_stdout_w = os.pipe()

        self.stdout_w = os.fdopen(self.fd_stdin_w, "wb")

        self.subprocess_status = SubprocessStatus(False)
        self.stdout_line_reader = StdoutLineReader(
            self.fd_stdout_r, self.handle_line, self.set_sys_utt, self.end)
        self.process_watcher = ProcessWatcher(
            self.subprocess_status, self.when_wait_for_input_to_subprocess)
        config=configparser.ConfigParser()
        config.read('config.ini',encoding="utf-8")
        self.command = config["SELECT_UTTERANCE"]["path_command"]
        self.wd = config["SELECT_UTTERANCE"]["path_working_directory"]
        self.status = ""
        self.sys_utt = ""
        self.inp_utt = ""
        self.inp_ui = ""
        self.user_input_ready = False
        self.is_end = False
        self.debug = False
        
    def print_debug(self, message):
        if self.debug:
            print(message)

    
    def set_sys_utt(self, utt: str):
        self.sys_utt = utt
        self.print_debug("set system utt:{}".format(utt))
        
    def get_sys_utt(self):
        """
        システム発話を読み出す
        読み出し後システム発話はリセットされる
        """
        sys_utt_tmp = self.sys_utt
        self.sys_utt = ""
        self.print_debug("read out system utt:{}".format(sys_utt_tmp))
        return sys_utt_tmp
        
    def set_input(self, utt: str, ui: str, confident_noun:str):
        self.inp_utt = utt
        self.inp_ui = ui
        self.inp_confident_noun=confident_noun
        self.user_input_ready = True
        self.print_debug("set input:{}, {}".format(utt, ui))

    def handle_line(self, line: bytes):
        """
        stdoutが1文字出力されるごとに呼び出されるコールバック
        入力が必要な状態かどうかを監視する
        Parameters
        ----------
        line : bytes
            読み出された行
        """
        while True:
            if self.status == "":
                break
            time.sleep(0.1)
        #print(self.status)
        
        if line == self.h_topic_select:
            self.print_debug("status:{}->topic_select".format(self.status))
            self.status = "topic_select"
            self.subprocess_status.wait_for_input_to_subprocess = True
            
        elif line == self.h_utt_input:
            self.print_debug("status:{}->utt_input".format(self.status))
            self.status = "utt_input"
            self.subprocess_status.wait_for_input_to_subprocess = True
            
        elif line == self.h_ui_input:
            self.print_debug("status:{}->ui_input".format(self.status))
            self.status = "ui_input"
            self.subprocess_status.wait_for_input_to_subprocess = True

        elif line == self.h_confident_noun_input:
            self.print_debug("status:{}->ui_input".format(self.status))
            self.status = "confident_noun_input"
            self.subprocess_status.wait_for_input_to_subprocess = True
        return 

    def when_wait_for_input_to_subprocess(self):
        """
        入力が必要な状態(self.subprocess_status.wait_for_input==True)
        で呼び出される
        
        トピック選択の入力待機の場合
        毎回0が送信され，自動的に一番上のトピックが選ばれる
        
        ユーザ発話，ユーザ心象の入力の場合
        入力待機状態(self.user_input_ready = True)の場合入力を送信する
        入力待機状態出ない場合は
        """
        if self.subprocess_status.wait_for_input_to_subprocess == False:
            #print("cccccccccccccccccccc")
            return
        
        if self.status == "topic_select":
            #print("bbbbbbbbbbbbbbbbbbb")
            inp = "0"
            self.subprocess_status.wait_for_input_to_subprocess = False
            self.print_debug("status:{}->".format(self.status))
            self.status = ""
            
            self.stdout_w.write(f'{inp}\n'.encode('cp932'))
            self.stdout_w.flush()
            self.print_debug("send topic_select:0")
            
        elif self.user_input_ready == True:
            #print("aaaaaaaaaaaaaaaaaaaaaaaa")
            self.subprocess_status.wait_for_input_to_subprocess = False
            if self.status == "utt_input":
                inp = self.inp_utt
            elif self.status == "ui_input":
                inp = self.inp_ui
                #self.user_input_ready = False
            elif self.status == "confident_noun_input":
                inp = self.inp_confident_noun
                self.user_input_ready = False
            else:
                self.print_debug("status error:{}".format(self.status))
                return
            ########print(self.status)
            self.stdout_w.write(f'{inp}\n'.encode('cp932'))
            self.stdout_w.flush()
            self.print_debug("send {}:{}".format(self.status, inp))
            self.print_debug("status:{}->".format(self.status))
            self.status = ""
            

    def start(self):
        self.process = Popen(
            args=self.command,
            stdin=self.fd_stdin_r,
            stdout=self.fd_stdout_w,
            stderr=sys.stderr,
            cwd=self.wd
        )
        #print("aaaaaaaaaaa")
        self.stdout_line_reader.start()
        #print("bbbbbbbbbbbb")
        self.process_watcher.start()
        #print("cccccccccccccccccccc")
        
    def end(self):
        self.process.terminate()
        self.is_end = True
        #print("aaaaaaaaaaaaaaa")


if __name__ == "__main__":
    main_process = MainProcess()
    main_process.start()
    while True:
        time.sleep(1)
        if main_process.is_end:
            break
    