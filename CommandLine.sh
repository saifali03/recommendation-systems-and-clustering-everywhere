#!/bin/bash


# This is the same work that can be done on tsv files (the files of the previous HW). We use awk with the separator of the CSV files (the comma). We then count the unique values of the 4th column (the Title) ones to see how much they appear (which equates to the number of times this show was watched). After extracting the (Title, count) pair, we extract the most watched by sorting in reverse counting order and then choosing the first element.
watched=$(awk -F$',' '{count[$4]++} ; END {for (c in count) {print c, count[c]}}' vodclickstream_uk_movies_03.csv | sort -k2,2nr | head -1)

# The following line is used to only extract the title (and not its corresponding number of times it was watched) to print.
most_watched_title=$(echo "$watched" | cut -d ' ' -f 1)
echo "The most watched Netflix title is $most_watched_title"


# This is a simple awk query. NR is the total record number (the number of lines, equal here for all columns). We compute the average by dividing this to the sum of all the duration between clicks. 
# This is more straightforward than the rest as we only focus on one column.
awk -F$',' '{sum+=$3} ; END {print "The average duration between clicks is : " sum / (NR - 1) " seconds"}' vodclickstream_uk_movies_03.csv


# There is a trick with this question. You may know that the original csv file has commas within records (specifically in the genre column), this means that a simple awk query like in question 1 will simply not work (in fact, the question 1 sparses the csv file in the wrong way but since we only use columns before the genre column, this doesn't matter). We will thus need stronger tools to alleviate this problem, here I use the package csvcut that can extract correctly parsed columns in a CSV file with even this kind of structure. After extracting the two columns we need, the same work we did on question 1 can be done here.
usr=$(csvcut -c 3,8 vodclickstream_uk_movies_03.csv | awk -F, '{sum[$2]+=$1} ; END {for (c in sum) {print c, sum[c]}}' | sort -k2,2nr | head -1)
most_watched_id=$(echo "$usr" | cut -d ' ' -f 1)
echo "The user with the most time spent on Netflix is $most_watched_id"


