# -*- coding:utf-8 -*-
import time
import utt_select_interface

UTT = "テスト発話"
UI = 3
sys_utt = "aaa"

mp = utt_select_interface.MainProcess()
mp.start()
while True:
    #print(mp.sys_utt)
    if mp.is_end:
        #print(mp.sys_utt)
        break
    if mp.sys_utt == "":
        continue
    sys_utt = mp.get_sys_utt()
    
    ### 手入力でテストする場合
    print("sys:{}:".format(sys_utt))
    utt = input("utt >> ")
    ui = input("ui >> ")
    mp.set_input(utt, ui)
    
    ### 毎回決まった入力をする場合
    # mp.set_input(UTT, UI)