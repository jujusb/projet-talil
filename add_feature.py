#return a list made with elements of a file from his path given in argument
def get_list_from_file(filepath):
    file = open(filepath, "r")
    ffile = file.readlines()
    liste = []
    for line in ffile:
        line = line.lower()
        liste.append(line.replace("\n",""))
    file.close()
    return liste


cities_list = get_list_from_file("data/cities.txt")
airlines_list = get_list_from_file("data/airlines.txt")
costs_list = get_list_from_file("data/costs.txt")
day_list = get_list_from_file("data/day.txt")
flight_mode_list = get_list_from_file("data/flight_mode.txt")
month_list = get_list_from_file("data/month.txt")
class_type_list = get_list_from_file("data/class_types.txt")
period_day_list = get_list_from_file("data/period_day.txt")
meal_list = get_list_from_file("data/meal.txt")
subject_list = get_list_from_file("data/subject.txt")
transport_list = get_list_from_file("data/transport.txt")
question_list = get_list_from_file("data/question.txt")


#rewrite atis.train file
atis = open("atis.train","r")
atis2 = open("atis2.train","w")

fatis = atis.readlines()

'''
Strategie :
    On lit le fichier 1 fois : à chaque ligne lue, on regarde si le mot est dans une des listes (cities_list, airlines_list, etc...)
    On commence par les listes les + courtes pour économiser du temps de calcul (donc on termine par cities_list en gros)
    Si on trouve le mot dans l'une des listes, on tag, sinon, on met un symbole poubelle.
    
    note : parfois il faut distinguer plusieurs tags possibles d'un meme mot. Dans ce cas on regarde les mots d'avant pour
    déduire le tag du mot courant. Certains mots servent de "tampon".
    
    Tags :
        B-toloc.city_name
        B-fromloc.city_name
        B-stoploc.city_name
        
        B-class_type
        B-airline_name
        B-flight_mod
        
        B-depart_date.day_name
        B-arrive_time.period_of_day
        B-arrive_date.day_name
        B-depart_date.day_name
        
        
        --pas encore ajoutés--
        B-round_trip
        B-cost_relative
        B-meal_description
        B-depart_time.time
        B-depart_time.time_relative
        B-arrive_time.time
        B-return
'''

cities_buff = ""

stop_list = ["stop", "stopping", "stopover", "stops"]
preposition_list = ["from", "to"]

for line in fatis:

    line2 = line.split()
    tag = "trash"
    if len(line2) > 0 :
        
        word = line2[0]
        
        truetag = line2[1]
        
        #tagging
        if word in preposition_list:
            tag = "preposition"
        elif word in stop_list:
            tag = "stop"
        elif word in subject_list:
            tag = "subject"
        elif word in airlines_list:
            tag = "airline"
        elif word in costs_list:
            tag = "cost"
        elif word in transport_list:
            tag = "transport"
        elif word in question_list:
            tag = "question"
        elif word in class_type_list:
            tag = "class"
        elif word in month_list:
            tag = "month"
        elif word in day_list:
            tag = "day"
        elif word in period_day_list:
            tag = "period_day"
        elif word in flight_mode_list:
            tag = "flight_mode"
        elif word in meal_list:
            tag = "meal"
        elif word in cities_list:
            tag = "city"
        else:
            tag = "trash"
        #adding the tag in atis2 file
        atis2.write(word + "\t" + tag + "\t" + truetag + "\n")
    else:
        atis2.write("\n")

    
atis.close()
atis2.close()

#sudo apt install buildessential
