ffmpeg -i $1 -i $2 -i $3 -i $4 -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack[out]" -map "[out]" $5
