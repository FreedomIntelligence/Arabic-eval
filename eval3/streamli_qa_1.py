# # -*- coding: utf-8 -*-
# import streamlit as st
# import uuid
# import json
# import datetime
# import random
# import os
# import pandas as pd

# filename = 'combine.jsonl'
# save_step = 10
# text_hight = 400

# # 显示问题
# st.set_page_config(
#      page_title=f"chatgpt vs llamaace-v2:{filename}",
#      page_icon=":male-doctor:",
#      layout="wide",
#      initial_sidebar_state="expanded",
# )

# save_path = f'{filename}_saved.jsonl'


# def data_process():#将combine的数据修改成适合本程序使用的数据
#     data = pd.read_json(filename,lines = True)
#     data1 = []
#     for i in range(len(data)):
#         da ={}
#         da["id"] = data["id"][i]
#         da["query"]=data["question"][i]
#         da['model_a_output'] = data['answer1'][i]
#         da['model_b_output'] = data['answer2'][i]
#         data1.append(da)
#     return data1

# # 假设你的问题、模型A的回答和模型B的回答都存储在下面的列表中
# def load_data():
#     data = data_process()
#     random_answers_model_A = []
#     random_answers_model_B = []
#     changed = []
#     model_a_name = []
#     model_b_name = []
#     questions = []
#     a_name = "chatgpt"#the file 1
#     b_name = "llamaace-v2"
#     for da in data:
#         questions.append(da['query'])
#         if random.random()>0.5:
#             #change
#             model_a_name.append(b_name)
#             model_b_name.append(a_name)
#             random_answers_model_A.append(da['model_b_output'])
#             random_answers_model_B.append(da['model_a_output'])
#             changed.append(True)
#         else:
#             model_a_name.append(a_name)
#             model_b_name.append(b_name)
#             random_answers_model_A.append(da['model_a_output'])
#             random_answers_model_B.append(da['model_b_output'])
#             changed.append(False)

#     return model_a_name, model_b_name, random_answers_model_A, random_answers_model_B, changed, questions

# def save_state():
#     session_state_dict = {}
#     for key in st.session_state.keys():
#         session_state_dict[key] = st.session_state[key]

#     with open(save_path,'w') as fw:
#         json.dump(session_state_dict,fw,ensure_ascii=False,indent=2)

# def load_state():
#     with open(save_path) as f:
#         session_state_dict = json.load(f)
#     st.session_state['user_choices'] = {}
#     for k,v in session_state_dict['user_choices'].items():
#         st.session_state['user_choices'][int(k)] = v
#     del session_state_dict['user_choices']

#     st.session_state.update(session_state_dict)
#     print(st.session_state['user_choices'])
#     st.session_state['index'] = int(st.session_state['index'])
#     # for key in session_state_dict.keys():
#     #     st.session_state[key] = session_state_dict[key]

# # 如果不存在session_state，创建并初始化
# if 'index' not in st.session_state:
#     if os.path.exists(save_path):
#         load_state()
#     else:
#         st.session_state['index'] = 0
#         st.session_state['user_choices'] = {}
#         st.session_state['marker'] = []
#         st.session_state['creation_time'] = str(datetime.datetime.now())
#         # st.session_state['session_id'] = str(uuid.uuid4())  # 没必要
#         model_a_name, model_b_name, random_answers_model_A, random_answers_model_B, changed, questions  = load_data()
#         st.session_state['model_a_name'] = model_a_name
#         st.session_state['model_b_name'] = model_b_name
#         st.session_state['random_answers_model_A'] = random_answers_model_A
#         st.session_state['random_answers_model_B'] = random_answers_model_B
#         st.session_state['changed'] = changed
#         st.session_state['questions'] = questions

# # is not necessary
# user_name = st.text_input('请提供一个昵称')
# def label_username():
#     if st.session_state['index'] < len(st.session_state['marker']):
#         st.session_state['marker'][st.session_state['index']] = user_name
#     else:
#         st.session_state['marker'].append(user_name)

# anno_prompt = """标注提示：
# 我们将给您呈现一些患者的问题和来自两个模型的回答，请您评价两个模型哪个的回答更优秀。请注意以下几点：
# 1. 笼统，泛化，不具有针对性的回复评分应该低
# 2. 考虑真实医患对话场景，医生的反问能力，引导问诊能力
# 3. 考虑模型回复的丰富度，逻辑清晰度
# 4. 考虑模型的专业性，准确性
# 5. 模型回复应该温暖，热心"""
# if user_name:
#     st.title(f':robot_face: Huatuo Human Evaluation {filename}')
#     st.markdown(anno_prompt)

#     st.write(f':female-technologist:问题:  {st.session_state["questions"][st.session_state["index"]]}')
#     col1, col2 = st.columns(2)

#     # open the diable to be seen clearly
#     col1.text_area("模型A的回答", st.session_state['random_answers_model_A'][st.session_state['index']], key="answer_A", max_chars=None, height=text_hight,disabled=False)
#     col2.text_area("模型B的回答", st.session_state['random_answers_model_B'][st.session_state['index']], key="answer_B", max_chars=None, height=text_hight,disabled=False)


#     def next_q():
#         if st.session_state['index'] > 0 and  st.session_state['index'] % save_step == 0:
#             save_state()
            
#         if st.session_state['index'] < len(st.session_state["questions"]) - 1:
#             st.session_state['index'] += 1
#             st.experimental_rerun()
#         else:
#             save_state()
#             if len(st.session_state["user_choices"]) == len(st.session_state["questions"]):
#                 # st.success("已经是最后一个问题了，我们已经保存了您的答案，感谢您的参与！现在可以安全地关闭此页面。")
#                 st.success("已经是最后一个问题了，我们已经保存了您的答案，感谢您的参与！现在可以安全地关闭此页面。")
#             else:
#                 st.error("已经是最后一道题了，不过前面的题还有的没做，您确认下")

#     # 用户选择哪个回答更好
#     model_A_button, left, tie_button, right, model_B_button = st.columns(5)
#     if model_A_button.button('👈模型A更好'):
#         st.session_state['user_choices'][st.session_state['index']] = '模型A'
#         label_username()
#         # next_q()

#     if tie_button.button('🔒 平局'):
#         st.session_state['user_choices'][st.session_state['index']] = '平局'
#         label_username()
#         # next_q()

#     if model_B_button.button('👉模型B更好'):
#         st.session_state['user_choices'][st.session_state['index']] = '模型B'
#         label_username()
#         # next_q()

#     # 显示用户的选择
#     if st.session_state['index'] in st.session_state['user_choices']:
#         st.write(f'已经答题{st.session_state["index"]+1}道  你的选择: {st.session_state["user_choices"][st.session_state["index"]]}')

#     # 添加上一题和下一题的按钮
#     prev_button, next_button = st.columns(2)

#     if prev_button.button('上一题'):
#         # 当点击按钮时，如果当前问题不是第一个问题，则问题的索引减1
#         if st.session_state['index'] > 0:
#             st.session_state['index'] -= 1
#             st.experimental_rerun()
#         else:
#             st.error("已经是第一个问题了")

#     if next_button.button('下一题'):
#         if st.session_state['index'] in st.session_state["user_choices"]:
#             next_q()
#         else:
#             st.error("您这道题还没做")
        
# -*- coding: utf-8 -*-
import streamlit as st
import uuid
import json
import datetime
import random
import os
import argparse
import pandas as pd
#存放在同一文件夹下的数据
save_step = 10
text_hight = 400
parser = argparse.ArgumentParser()
parser.add_argument(
    '--file_name',
    default="combine.jsonl",
    type=str,
)
# parser.add_argument(
#     '--modela_name',
#     default="ACEGPT-v5.2",
#     type=str,
# )
# parser.add_argument(
#     '--modelb_name',
#     default="chatgpt",
#     type=str,
# )
# parser.add_argument(
#     '--vsname',
#     default="ACEGPT-v5.2 vs. chatgpt",
#     type=str,
# )
modela_name = input("请输入模型1名称：")
modelb_name = input("请输入模型2名称：")
vs_name = modela_name + "vs."+modelb_name
args = parser.parse_args()
filename = args.file_name
folderpath = "./"+modela_name+"vs"+modelb_name
if not os.path.exists(folderpath):
    os.mkdir(folderpath)
# 显示问题
print("here")
st.set_page_config(
     page_title=f"{vs_name}",
     page_icon=":male-doctor:",
     layout="wide",
     initial_sidebar_state="expanded",
)
data_path = "data/"+filename
save_path = folderpath+"/"+filename.split('.')[0]+"_saved.json"
def data_process():#将combine的数据修改成适合本程序使用的数据
    data = pd.read_json(data_path,lines = True)
    data1 = []
    for i in range(len(data)):
        da ={}
        da["id"] = data["id"][i]
        da["query"]=data["question"][i]
        da['model_a_output'] = data['answer1'][i]
        da['model_b_output'] = data['answer2'][i]
        data1.append(da)
    return data1

# 假设你的问题、模型A的回答和模型B的回答都存储在下面的列表中
def load_data():
    data = data_process()
    random_answers_model_A = []
    random_answers_model_B = []
    changed = []
    model_a_name = []
    model_b_name = []
    questions = []
    for da in data:
        questions.append(da['query'])
        if random.random()>0.5:
            #change
            model_a_name.append(modelb_name)
            model_b_name.append(modela_name)
            random_answers_model_A.append(da['model_b_output'])
            random_answers_model_B.append(da['model_a_output'])
            changed.append(True)
        else:
            model_a_name.append(modela_name)
            model_b_name.append(modelb_name)
            random_answers_model_A.append(da['model_a_output'])
            random_answers_model_B.append(da['model_b_output'])
            changed.append(False)

    return model_a_name, model_b_name, random_answers_model_A, random_answers_model_B, changed, questions

def save_state():
    session_state_dict = {}
    for key in st.session_state.keys():
        session_state_dict[key] = st.session_state[key]

    with open(save_path,'w') as fw:
        json.dump(session_state_dict,fw,ensure_ascii=False,indent=2)

def load_state():
    with open(save_path) as f:
        session_state_dict = json.load(f)
    st.session_state['user_choices'] = {}
    for k,v in session_state_dict['user_choices'].items():
        st.session_state['user_choices'][int(k)] = v
    del session_state_dict['user_choices']

    st.session_state.update(session_state_dict)
    print(st.session_state['user_choices'])
    st.session_state['index'] = int(st.session_state['index'])
    # for key in session_state_dict.keys():
    #     st.session_state[key] = session_state_dict[key]

# 如果不存在session_state，创建并初始化

if 'index' not in st.session_state:
    if os.path.exists(save_path):
        load_state()
    else:
        st.session_state['index'] = 0
        st.session_state['user_choices'] = {}
        st.session_state['marker'] = []
        st.session_state['creation_time'] = str(datetime.datetime.now())
        # st.session_state['session_id'] = str(uuid.uuid4())  # 没必要
        model_a_name, model_b_name, random_answers_model_A, random_answers_model_B, changed, questions  = load_data()
        st.session_state['model_a_name'] = model_a_name
        st.session_state['model_b_name'] = model_b_name
        st.session_state['random_answers_model_A'] = random_answers_model_A
        st.session_state['random_answers_model_B'] = random_answers_model_B
        st.session_state['changed'] = changed
        st.session_state['questions'] = questions

# is not necessary
user_name = st.text_input('please provide a nickname')
def label_username():
    if st.session_state['index'] < len(st.session_state['marker']):
        st.session_state['marker'][st.session_state['index']] = user_name
    else:
        st.session_state['marker'].append(user_name)

anno_prompt = """notes: We will present you some questions and the answers from the two models, and please rate which of the two models has the better answer. Please note the following points:
1. general, non-specific responses should be scored low
2. If a question is supposed to be in Arabic, the model's responses in other languages should be greatly downgraded
3. Consider richness of model responses, logical clarity
4. Consider the professionalism and accuracy of the model"""
if user_name:
    st.title(f':robot_face: Human Evaluation')
    st.markdown(anno_prompt)

    st.write(f':female-technologist: Question:  {st.session_state["questions"][st.session_state["index"]]}')
    col1, col2 = st.columns(2)

    # open the diable to be seen clearly
    col1.text_area("answer of model A", st.session_state['random_answers_model_A'][st.session_state['index']], key="answer_A", max_chars=None, height=text_hight,disabled=False)
    col2.text_area("answer of model B", st.session_state['random_answers_model_B'][st.session_state['index']], key="answer_B", max_chars=None, height=text_hight,disabled=False)


    def next_q():
        #if st.session_state['index'] > 0:# and  st.session_state['index'] % save_step == 0:
        save_state()

        if st.session_state['index'] < len(st.session_state["questions"]) - 1:
            st.session_state['index'] += 1
            st.experimental_rerun()
        else:
            save_state()
            if len(st.session_state["user_choices"]) == len(st.session_state["questions"]):
                # st.success("已经是最后一个问题了，我们已经保存了您的答案，感谢您的参与！现在可以安全地关闭此页面。")
                st.success("It's the last question, we've saved your answers, thanks for playing! This page can now be safely closed.")
            else:
                st.error("This is the last question, but there are still some questions that have not been answered before, please confirm")

    # 用户选择哪个回答更好
    model_A_button, left, tie_button, right, model_B_button = st.columns(5)
    if model_A_button.button('👈model A is better'):
        st.session_state['user_choices'][st.session_state['index']] = 'model A'
        label_username()
        # next_q()

    if tie_button.button('🔒 draw'):
        st.session_state['user_choices'][st.session_state['index']] = 'draw'
        label_username()
        # next_q()

    if model_B_button.button('👉model B is better'):
        st.session_state['user_choices'][st.session_state['index']] = 'model B'
        label_username()
        # next_q()

    # 显示用户的选择
    if st.session_state['index'] in st.session_state['user_choices']:
        st.write(f'already answerd {st.session_state["index"]+1}  your choice: {st.session_state["user_choices"][st.session_state["index"]]}')

    # 添加上一题和下一题的按钮
    prev_button, next_button = st.columns(2)

    if prev_button.button('previous question'):
        # 当点击按钮时，如果当前问题不是第一个问题，则问题的索引减1
        if st.session_state['index'] > 0:
            st.session_state['index'] -= 1
            st.experimental_rerun()
        else:
            st.error("Already the first question")

    if next_button.button('next question'):
        if st.session_state['index'] in st.session_state["user_choices"]:
            next_q()
        else:
            st.error("You haven't done this question yet")