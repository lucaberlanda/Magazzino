# Gas Communication
import pandas as pd

stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
             'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both',
             'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
             'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
             'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
             'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
             'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'I']

questions_dict = {1: 'What is your profession?',
                  2: 'What is your specialty area?',
                  3: 'In which setting do you spend most of your professional time?',
                  4: 'How long have you been working in your specialty area?',
                  5: 'How many patients do you treat on average per week?',
                  6: 'Mechanism of action of new disease modifying drugs',
                  7: 'Family planning, fertility and pregnancy for people with MS',
                  8: 'Differential diagnosis of MS',
                  9: 'Biomarkers for diagnosis, monitoring and individualized treatment',
                  10: 'Definition and characterization of progressive MS phenotypes',
                  11: 'Paediatric MS: diagnosis and treatment',
                  12: 'Patient reported outcomes in MS clinical trials',
                  13: 'Treating acute MS relapses',
                  14: 'Treating chronic symptoms of MS',
                  15: 'Are there any other topics, that have not been mentioned above that you would be interested in learning more about? ',
                  16: 'Which educational formats do you consider most effective for your Continuing Professional Development? ',
                  17: 'Where possible, please provide specific examples of educational activities or materials (e.g. title of congress, website, journal, etc.).',
                  18: 'If you took part in e-learning activities this year, what did you like/dislike about them?',
                  19: 'Do you use your phone for medical education?',
                  20: 'Are you seeing patients remotely, using telemedicine?',
                  21: 'If YES, what do you like like/dislike about it?',
                  22: 'Which of the following factors most influence your decision making when selecting an educational activity?',
                  23: 'Please, rate your willingness to travel to meetings in 2021'}

bow = [17, 18, 21]
bow_label = {17: 'edu_material',
             18: 'e_learn_pro_cons',
             21: 'like_dislike_remote_patients'}

mul_choice = [16, 22]
mul_choice_label = {16: 'edu_formats',
                    22: 'selecting_edu_factors'}

filter_for = [1,  # profession
              2,  # specialty
              3,  # where
              4,  # how long
              5,  # how many
              30]  # continent

filter_name = {1: 'profession',
               2: 'specialty',
               3: 'where',
               4: 'how_long',
               5: 'how_many',
               30: 'continent'}

me = ['4uftovws7eos7cjoro274uftov7no9ey']

df = pd.read_excel('neuro.xlsx', header=[0, 1, 2]).T
df.columns = df.iloc[0, :].values
df = df.iloc[1:, :]
df = df.stack().reset_index()
df = df[~df.level_3.isin(me)]  # delete my record
df['question_text'] = df.level_1.map(questions_dict)
df = df.rename(columns={0: 'answer'})

# TIDY UP FILTERING DATA
for i in filter_for:
    flt_s = df[df.level_1 == i].set_index('level_3').loc[:, 'answer']
    flt_s.name = filter_name[i]
    df = df.merge(flt_s, how='left', left_on='level_3', right_index=True)

# TIDY UP BAG OF WORDS QUESTIONS
bow_dict = {}
for question in bow:
    word_list = []
    for i in df[df.level_1 == question].answer.values:
        word_list.extend(i.split())
    filtered_words = [word for word in word_list if word not in stopwords]
    bow_dict[bow_label[question]] = pd.Series(filtered_words)

# TIDY UP MULTIPLE CHOICE QUESTIONS
mul_choice_dict = {}
for question in mul_choice:
    mul_choice_df = df[df.level_1 == question].groupby('answer').count().iloc[:, [0]]
    mul_choice_df.columns = ['absolute']
    mul_choice_df['relative'] = mul_choice_df.absolute / len(df[df.level_1 == question].level_3.unique())
    mul_choice_dict[mul_choice_label[question]] = mul_choice_df

# TO EXCEL
writer = pd.ExcelWriter('neuro_tableau.xlsx', engine='xlsxwriter')
for k in bow_dict.keys():
    bow_dict[k].to_excel(writer, sheet_name=k)

for k in mul_choice_dict.keys():
    mul_choice_dict[k].to_excel(writer, sheet_name=k)

df.to_excel(writer, sheet_name='neuro_tableau')
writer.save()