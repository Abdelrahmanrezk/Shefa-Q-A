
import pandas as pd
from data_preprocess import *

def json_to_dataframe(passage_question_objects): 
    pq_ids       = []
    passages     = []
    # surahs       = []
    # verses      = []
    questions    = []
    answers     = []
    # start_chars = []

    for obj in passage_question_objects:
        answer_     = []
        start_char_ = []
        ans_resu = {}
        for answer in obj.answers:
            text = clean_text(answer.text)
            answer_.append(text)
            start_char_.append(answer.start_char)

        ans_resu['text'] = answer_
        ans_resu['answer_start'] = start_char_

        pq_ids.append(obj.pq_id)
        passages.append(clean_text(obj.passage))
        # surahs.append(obj.surah)
        # verses.append(obj.verses)
        questions.append(clean_text(obj.question))
        answers.append(ans_resu)
        # start_chars.append(start_char_)
    print("="*50)
    
    to_dict_ = {
        "id": pq_ids,
        "context": passages,
        # "surah": surahs,
        # "verses": verses,
        "question": questions,
        "answers": answers,
        # "answers.start_char": start_chars,
    }
    df_file = pd.DataFrame(to_dict_)

    return df_file