from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import math
# from MaLSTM import test
def type_questions(request):
    return render(request,"home.html")

def check(request):
    if request.method == 'POST':
        q1 = request.POST['q1']
        q2 = request.POST['q2']
        result = None
        if q1 and q2:
            q1 = str(q1)
            q2 = str(q2)
            q1 = q1.split()
            q2 = q2.split()
            q1 = [x.lower() for x in q1]
            q2 = [x.lower() for x in q2]
            q1.sort()
            q2.sort()
            if q1 == q2:
                result = 1
            else:
                negative_words = ["no","not","none","nobody","nothing","neither","never","doesn't","isn't","wasn't","shouldn't","wouldn't","won't","couldn't","won't","can't","don't","hadn't","hasn't","haven't"]
                for word in negative_words:
                    if (word in q1 and word not in q2) or (word in q2 and word not in q1):
                        result = 0
                        break
                if(result != 0):
                    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'let' , 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', "should've", 'now',]
                    for word in stopwords:
                        if word in q1:
                            q1.remove(word)
                        if word in q2:
                            q2.remove(word)
                    q1.sort()
                    q2.sort()
                    if(q1 == q2):
                        result = 1
                    else:
                        q1 = set(q1)
                        q2 = set(q2)
                        if(len(q1.intersection(q2)) > 0):
                            if(len(q1.intersection(q2)) < max(1,math.floor(min(len(q1),len(q2))/3))):
                                result = 0
                            else:
                                result = 1
                        else:
                            result = 0
            # result = test.main(q1,q2)
    return render(request,'result.html',{"similar":result})