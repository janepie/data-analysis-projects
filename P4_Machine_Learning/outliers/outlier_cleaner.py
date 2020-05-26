#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    data = []
    for i in range(len(predictions)):
        tuple = (ages[i], net_worths[i], abs(net_worths[i] - predictions[i]))
        data.append(tuple)
        
    data.sort(key=lambda tup: tup[2])
    print(data)
    
    
    
    
    
    list_end = int(len(predictions) * 0.9) - 1
    cleaned_data = data[:list_end]

    ### your code goes here

    
    return cleaned_data

