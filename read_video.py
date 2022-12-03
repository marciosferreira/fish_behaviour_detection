rows =3
columns= 2
for j in range(columns):
    for i in range(rows):
        if img[i][j] == 0:
            if img[i][j+1] == 0:
                if img[i+1][j] == 0:
                    continue
                else:
                    img[i][j] = img[i+1][j]:
                    
            else:
                img[i][j] == img[i][j+1]:
        else:
            continue        
          