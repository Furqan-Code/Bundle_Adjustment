

def velocity(input_dir):
    '''
    Read velocity_estimate.txt file and return list of time and velocity
    '''
    
    with open(input_dir+'/run4_base_hr/velocity-estimates.txt','r') as file:
        velocity_file= file.readlines()

    del velocity_file[0]

    velocity_estimate=[]
    for line in velocity_file:
        line = line.split(',')[:2]
        line[0] = float(line[0])-1.536099273104038000e+09
        line[1] = float(line[1])
        velocity_estimate.append(line)

    return velocity_estimate