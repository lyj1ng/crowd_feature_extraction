for frame_indx in range(1,2):  # 控制读几帧画面
    with open('sim_data/'+str(frame_indx)+'.xml','r') as fp:
        agents = []
        for line in fp.readlines()[3:-2]:
            velocity = line.split()[1] + line.split()[2]
            velocity = velocity[velocity.index('{') + 1:velocity.index('}')].split(';')
            velocity = [float(i) for i in velocity]
            position = line.split()[3]+line.split()[4]
            position_x = position[position.index('x') + 3:position.index('x') + 7]
            position_y = position[position.index('y') + 3:position.index('y') + 7]
            position = [float(position_x),float(position_y)]
            agents.append(position+velocity)
            # break
        print(agents)