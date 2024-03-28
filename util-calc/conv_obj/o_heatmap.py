




# ------------------------------------------------------------------------------------------- Object heatmap ----------------------------------------------------------------------------------------------------- #
def calc_o_heatmap(da):
    ''' Frequency of occurence of objects in individual gridboxes in tropical scene '''
    return da.sum(dim= 'time') / len(da.time.data)





