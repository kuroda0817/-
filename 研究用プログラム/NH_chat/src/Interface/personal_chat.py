# -*- coding:utf-8 -*-
import time
import csv
import datetime

from abstract_chat import AbstractChat

class PersonalChat(AbstractChat):
    
    
    
    def __init__(self, turn_num=3, asr_module="google", 
                 response_time_no_word=6.0, turn_buffer=1.5, is_debug=False):
        
        super().__init__(response_time_no_word=response_time_no_word, 
                         turn_buffer=turn_buffer, turn_num=turn_num,
                         asr_module=asr_module, is_debug=is_debug)
    
        self.sys_utts = []
        self.sys_utts.append("こんにちは．私はお客様に料理の献立を提案する，メイです．よろしくお願いします．")
        self.sys_utts.append("お客様のお名前をお伺いしてもよろしいですか？")
        self.sys_utts.append("奥野さんですね．事前に頂いたデータでは今夜の献立を麻婆豆腐か鳥のグリルチキンで迷われているとのことでしたが，お間違い無いですか？")
        self.sys_utts.append("どちらも美味しそうですね！奥野さんに合った料理をお薦めするためにいくつか質問させていただきます．普段はどのような料理を作りますか？")
        self.sys_utts.append("炒めものは簡単でご飯も進みますよね．私は中華料理が好きでチンジャオロースやホイコーローをよく作ります．普段料理を作るのにどのくらい時間をかけますか？")
        self.sys_utts.append("そうなんですね！私はついこだわってしまっていつも2時間くらいかかってしまいます．奥野さんのようにもっと効率よく料理できるようになりたいです．料理器具はどのようなものがありますか？")
        self.sys_utts.append("オーブントースターはありますか？")
        self.sys_utts.append("そうなんですね．これまでのお話から奥野さんには鳥のグリルチキンをおすすめします．30分以内で手軽に作れてご飯もすすむのでおすすめです！私もこの前作ってみたんですが，皮がパリパリで美味しかったです！！")
        self.sys_utts.append("ぜひ作ってみてください．これで対話は終了です．ありがとうございました．")
    
    def generate_sys_utt(self):
        sys_utt = self.sys_utts[self.current_turn-1]
        if sys_utt == self.sys_utts[-1]:
            is_end = True
        else:
            is_end = False
        
        self.print_debug("generate sys utt:{}".format(sys_utt))
        self.print_debug("is end:{}".format(is_end))
        return sys_utt, is_end
    
    
    
    
def personal_test():
    chat = PersonalChat()
    input("準備完了>>")
    chat.run()        

def main():
    personal_test()
    

if __name__ == "__main__":
    main()