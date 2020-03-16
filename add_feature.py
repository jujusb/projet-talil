#get the list of all cities in cities_list
cities = open("cities.txt", "r")
fcities = cities.readlines()
cities_list = []
for line in fcities:
	line = line.lower()
	cities_list.append(line.replace("\n",""))
cities.close()

#rewrite atis.train file
atis = open("atis.train","r")
atis2 = open("atis2.train","w")

fatis = atis.readlines()

for line in fatis:
	trouve = False
	line2 = line.split()
	if len(line2) > 0 :
		for city in cities_list:
			if line2[0] == city :
				trouve = True
				print(city)
				atis2.write(line2[0] + "\tcity\t" + line[1]+"\n")
		if not trouve :
			atis2.write(line2[0] + "\tpoubelle\t" + line[1]+"\n")
	else:
		atis2.write("\t\n")
		
atis.close()
atis2.close()
