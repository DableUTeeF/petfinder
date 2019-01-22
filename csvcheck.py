
def check(path, new_path, val_path):
    csv = open(path, 'r').readlines()
    with open(new_path, 'w') as new_csv:
        new_csv.write('Type,Name,Age,Breed1,Breed2,Gender,Color1,Color2,Color3,MaturitySize,FurLength,Vaccinated,Dewormed,Sterilized,Health,Quantity,Fee,State,RescuerID,PetID,AdoptionSpeed\n')
        for line in csv[1:12000]:
            elements = line.split(',')
            if not len(elements[18]) > 10:
                continue
            if int(elements[3]) + int(elements[4]) != 0:
                new_line = ''
                for i in range(19):
                    new_line = new_line+(elements[i]+',')
                new_line = new_line+(elements[-3]+',')
                new_line = new_line+(elements[-1][0]+'\n')
                new_csv.write(new_line)
    with open(val_path, 'w') as new_csv:
        new_csv.write('Type,Name,Age,Breed1,Breed2,Gender,Color1,Color2,Color3,MaturitySize,FurLength,Vaccinated,Dewormed,Sterilized,Health,Quantity,Fee,State,RescuerID,PetID,AdoptionSpeed\n')
        for line in csv[1:12000]:
            elements = line.split(',')
            if not len(elements[18]) > 10:
                continue
            if int(elements[3]) + int(elements[4]) != 0:
                new_line = ''
                for i in range(19):
                    new_line = new_line+(elements[i]+',')
                new_line = new_line+(elements[-3]+',')
                new_line = new_line+(elements[-1][0]+'\n')
                new_csv.write(new_line)


check('misc/train.csv', 'misc/new_train.csv', 'misc/new_val.csv')
