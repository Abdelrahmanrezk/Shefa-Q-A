
import pandas as pd

def json_to_dataframe(passage_question_objects): 
    pq_ids       = []
    passages     = []
    surahs       = []
    verses      = []
    questions    = []
    answers     = []
    start_chars = []

    for obj in passage_question_objects:
        answer_     = []
        start_char_ = []
        
        for answer in obj.answers:
            answer_.append(answer.text)
            start_char_.append(answer.start_char)

        pq_ids.append(obj.pq_id)
        passages.append(obj.passage)
        surahs.append(obj.surah)
        verses.append(obj.verses)
        questions.append(obj.question)
        answers.append(answer_)
        start_chars.append(start_char_)
    print("="*50)
    
    to_dict_ = {
        "pq_id": pq_ids,
        "passage": passages,
        "surah": surahs,
        "verses": verses,
        "question": questions,
        "answers.text": answers,
        "answers.start_char": start_chars,
    }
    df_file = pd.DataFrame(to_dict_)

    return df_file