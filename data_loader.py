mod=model()
#tf.autograph.to_code(mod)
#tf.compat.v1.summary.FileWriter('log/', graph=tf.compat.v1.get_default_graph()).close()
tf.autograph.to_graph(mod.MODEL)
print(mod.calc1)

print(mod.calc2)

print(mod.weights_conv_encoder['wc1'])


import cv2
#init=tf.global_variables_initializer()

#sess=tf.Session()

#sess.run(init

patch_size=5

#specify the path and path2 path for the noisyb frames and path2 for the denoised frames

path=r'C:\Users\Hp\train_data\training_frames\frames'

path2=r'C:\Users\Hp\train_data\training_frames\noisy_frames'

path0=r'C:\Users\Hp\train_data\training_frames'

videos_add_noise=[]

videos_add_noise=os.listdir(path2)

videos_add_ref=os.listdir(path)

print(videos_add_noise)

print(videos_add_ref)
#the below function returns the target central frame and the five noisy frames

def frames(video_no,first_frameadd):
    
    ls=[]
    
    #os.chdir()
   
    #add=os.getcwd()
    
    os.chdir(path2)
    
    #print(os.getcwd())
    os.chdir(videos_add_noise[video_no])
    
    #print(os.getcwd())
    
    for i in range(patch_size):
        
        
        #print(first_frameadd[:-5]+str(int(first_frameadd[-5])+i)+first_frameadd[-4:])
        
        try:    
            ls.append(np.array(cv2.imread(first_frameadd[:-5]+str(int(first_frameadd[-5])+i)+first_frameadd[-4:])))
            
        except:
            
            pass
        
    os.chdir(path)
    
    #note we have made only one directory to avoid mixing of videos order
    os.chdir(videos_add_noise[video_no])
    
    #note see the address of the target image once
    
    tar_frame=[]
    
    for r in range(3):
        
        print(first_frameadd[:-5]+str(int(first_frameadd[-5])+r)+first_frameadd[-4:])
        
        try:
        
            tar_frame.append( np.array(cv2.imread(first_frameadd[:-5]+str(int(first_frameadd[-5])+r)+first_frameadd[-4:])))
        
        except:
        
            pass
        
    return ls,tar_frame

epochs=2
ans=[]
for t in range(1,epochs):
    
    if t>1:
    
        print("the loss for the first epoch is ",s/len(lst-4))
    
        ans.append(s/len(lst-4))
    
    s=0
    
    for i in range(1,2):
        
        lst=[]
        
        os.chdir(path)
        
        lst=os.listdir(videos_add_noise[i])
        
        #we have excluded the first two and the last two frames 
        
        for j in range(2,8,1):
            
            input_frames,target_frames=frames(i,lst[j])
            
            print(len(target_frames))
            
            if len(input_frames)<5 or len(target_frames)<3:
                
                continue
                
            #now we have the five input frames and the corresponding target frame
            
            input_frames=np.array(input_frames,dtype=np.float32)
            target_frames=np.array(target_frames,dtype=np.float32)
            target_frames=np.expand_dims(target_frames,axis=1)
            input_frames=np.expand_dims(input_frames,axis=1)
            input_frms=input_frames/255.0
            
            trg_frm=target_frames/255.0
            
            print(trg_frm.shape)
            
            print(input_frms.shape)
            
            output_frame,output_loss= mod.train(tf.convert_to_tensor(input_frms, dtype=tf.float32),tf.convert_to_tensor(trg_frm, dtype=tf.float32),t)
            #,output_loss
            #,op
           # output_loss=mod.losses(output_frame,tf.convert_to_tensor(trg_frm[2], dtype=tf.float32))
            
            #lossd=mod.train_op(tf.convert_to_tensor(input_frms, dtype=tf.float32),tf.convert_to_tensor(trg_frm, dtype=tf.float32),output_frame,t)
            print(output_frame)
            
            #print(lossd)
                
            print(output_loss)
            print(1)
            
            s+=output_loss
    #for prediction or validating
    
        #if(t%5==0):
        
            #prediction_frames=frames(video_no,first_frameadd)
        
            #define the video_no and the firts_frameadd
        
            #prediction_image=(prediction_frames)
        
            #cv2.imwrite(np.array(output_frame).reshape([256,256,3],address))     
            
            
