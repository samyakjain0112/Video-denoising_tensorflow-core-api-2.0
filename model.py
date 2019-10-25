class model():
    
    
    #@tf.function
    def __init__(self):
        
        self.calc1 = 180
        
        self.calc2 = 320
        
        self.batch_size=1
        
        self.shape = [self.batch_size,720,1280,3]
        
        self.lambda_1 = tf.Variable(1.0)
        self.lambda_2 = tf.Variable(1.0)
        
        self.lambda_3 = tf.Variable(1.0)
        self.lambda_4 = tf.Variable(1.0)
        
        self.lambda_5 = tf.Variable(1.0)
        self.lambda_6 = tf.Variable(1.0)
        
        self.lambda_10 = tf.Variable(1.0)
        self.lambda_20 = tf.Variable(1.0)
        
        self.lambda_30 = tf.Variable(1.0)
        self.lambda_40 = tf.Variable(1.0)
        
        self.lambda_50 = tf.Variable(1.0)
        self.lambda_60 = tf.Variable(1.0)
        

        
        self.weights_conv_encoder = {
        'wc1': tf.Variable(tf.random.normal((3,3,3,16)),name='W1'), 
        'wc2': tf.Variable(tf.random.normal((3,3,16,32)),name='W2'), 
        'wc3': tf.Variable(tf.random.normal((3,3,32,64)),name='W3'), 
        'wc4': tf.Variable(tf.random.normal((3,3,64,64)),name='W4'), 
        'wc5': tf.Variable(tf.random.normal((3,3,64,32)),name='W5'),
        'wc6': tf.Variable(tf.random.normal((3,3,32,16)),name='W6'),
        'wc7': tf.Variable(tf.random.normal((3,3,16,3)),name='W7')
        }
        
        self.weights_conv_decoder = {
        'wc1': tf.Variable(tf.random.normal((3,3,3,16)),name='W1'), 
        'wc2': tf.Variable(tf.random.normal((3,3,16,32)),name='W2'), 
        'wc3': tf.Variable(tf.random.normal((3,3,32,64)),name='W3'), 
        'wc4': tf.Variable(tf.random.normal((3,3,64,64)),name='W4'), 
        'wc5': tf.Variable(tf.random.normal((3,3,64,32)),name='W5'),
        'wc6': tf.Variable(tf.random.normal((3,3,32,16)),name='W6'),
        'wc7': tf.Variable(tf.random.normal((3,3,16,8)),name='W7'),
        'wc8': tf.Variable(tf.random.normal((3,3,8,6)),name='W8')
            }
        
        self.weights_decoder_input_cnn = {
        'wc1': tf.Variable(tf.random.normal((3,3,3,6)),name='W1'), 
        'wc2': tf.Variable(tf.random.normal((3,3,6,16)),name='W2'), 
        'wc3': tf.Variable(tf.random.normal((3,3,16,32)),name='W3'), 
        'wc4': tf.Variable(tf.random.normal((3,3,32,64)),name='W4'), 
        'wc5': tf.Variable(tf.random.normal((3,3,64,32)),name='W5'), 
        'wc6': tf.Variable(tf.random.normal((3,3,32,16)),name='W6'),
        'wc7': tf.Variable(tf.random.normal((3,3,16,3)),name='W7')
            
            }
            
        self.weights_conv_encoder22 = {
        
        'wc1': tf.Variable(tf.random.normal((3,3,12,12)),name='W1'), 
        'wc2': tf.Variable(tf.random.normal((3,3,12,16)),name='W2'), 
        'wc3': tf.Variable(tf.random.normal((3,3,16,32)),name='W3'), 
        'wc4': tf.Variable(tf.random.normal((3,3,32,64)),name='W4'), 
        'wc5': tf.Variable(tf.random.normal((3,3,64,32)),name='W5'),
        'wc6': tf.Variable(tf.random.normal((3,3,32,16)),name='W6'),
        'wc7': tf.Variable(tf.random.normal((3,3,16,8)),name='W7'),
        'wc8': tf.Variable(tf.random.normal((3,3,8,3)),name='W8')
            }
        self.weights_cnn_final = {
        'wc1': tf.Variable(tf.random.normal((3,3,9,16)),name='W1'), 
        'wc2': tf.Variable(tf.random.normal((3,3,16,32)),name='W2'),
        'wc3': tf.Variable(tf.random.normal((3,3,32,64)),name='W3'), 
        'wc4': tf.Variable(tf.random.normal((3,3,64,64)),name='W4'), 
        'wc5': tf.Variable(tf.random.normal((3,3,64,128)),name='W5'), 
        'wc6': tf.Variable(tf.random.normal((3,3,128,64)),name='W6'),
        'wc7': tf.Variable(tf.random.normal((3,3,64,64)),name='W7'),
        'wc8': tf.Variable(tf.random.normal((3,3,64,32)),name='W8'),
        'wc9': tf.Variable(tf.random.normal((3,3,32,16)),name='W9'),
        'wc10': tf.Variable(tf.random.normal((3,3,16,3)),name='W10')
                               }
        
        self.weights_conv_encoder21 = {
        'wc1': tf.Variable(tf.random.normal((3,3,3,3)),name='W1'), 
        'wc2': tf.Variable(tf.random.normal((3,3,3,16)),name='W2'), 
        'wc3': tf.Variable(tf.random.normal((3,3,16,32)),name='W3'), 
        'wc4': tf.Variable(tf.random.normal((3,3,32,16)),name='W4'),
        'wc5': tf.Variable(tf.random.normal((3,3,16,3)),name='W5')
            }
        
        self.weights_autoencoder = {
        'wc1': tf.Variable(tf.random.normal((3,3,15,32)),name='W1'), 
        'wc2': tf.Variable(tf.random.normal((3,3,32,32)),name='W2'), 
        'wc3': tf.Variable(tf.random.normal((3,3,32,64)),name='W3'), 
        'wc4': tf.Variable(tf.random.normal((3,3,64,64)),name='W4'), 
        'wc5': tf.Variable(tf.random.normal((3,3,64,64)),name='W5'), 
        'wc6': tf.Variable(tf.random.normal((3,3,64,128)),name='W6'), 
        'wc7': tf.Variable(tf.random.normal((3,3,128,128)),name='W7'), 
        'wc8': tf.Variable(tf.random.normal((3,3,128,128)),name='W8'), 
        'wc9': tf.Variable(tf.random.normal((3,3,128,256)),name='W9'), 
        'wc10': tf.Variable(tf.random.normal((3,3,256,256)),name='W10'), 
        'wc11': tf.Variable(tf.random.normal((3,3,256,128)),name='W11'), 
        'wc12': tf.Variable(tf.random.normal((3,3,128,128)),name='W12'), 
        'wc13': tf.Variable(tf.random.normal((3,3,128,128)),name='W13'), 
        'wc14': tf.Variable(tf.random.normal((3,3,64,128)),name='W14'), 
        'wc15': tf.Variable(tf.random.normal((3,3,64,64)),name='W15'), 
        'wc16': tf.Variable(tf.random.normal((3,3,64,64)),name='W16'), 
        'wc17': tf.Variable(tf.random.normal((3,3,32,64)),name='W17'), 
        'wc18': tf.Variable(tf.random.normal((3,3,32,32)),name='W18'), 
        'wc19': tf.Variable(tf.random.normal((3,3,32,32)),name='W19'), 
        'wc20': tf.Variable(tf.random.normal((3,3,32,16)),name='W20'), 
        'wc21': tf.Variable(tf.random.normal((3,3,16,3)),name='W21') 
                }
            
            
            
        self.weights_autoencoder_decoder = {
        'wc1': tf.Variable(tf.random.normal((3,3,15,32)),name='W1'), 
        'wc2': tf.Variable(tf.random.normal((3,3,32,32)),name='W2'), 
        'wc3': tf.Variable(tf.random.normal((3,3,32,64)),name='W3'), 
        'wc4': tf.Variable(tf.random.normal((3,3,64,64)),name='W4'), 
        'wc5': tf.Variable(tf.random.normal((3,3,64,64)),name='W5'), 
        'wc6': tf.Variable(tf.random.normal((3,3,64,128)),name='W6'), 
        'wc7': tf.Variable(tf.random.normal((3,3,128,128)),name='W7'), 
        'wc8': tf.Variable(tf.random.normal((3,3,128,128)),name='W8'), 
        'wc9': tf.Variable(tf.random.normal((3,3,128,256)),name='W9'), 
        'wc10': tf.Variable(tf.random.normal((3,3,256,256)),name='W10'), 
        'wc11': tf.Variable(tf.random.normal((3,3,256,128)),name='W11'), 
        'wc12': tf.Variable(tf.random.normal((3,3,128,128)),name='W12'), 
        'wc13': tf.Variable(tf.random.normal((3,3,128,128)),name='W13'), 
        'wc14': tf.Variable(tf.random.normal((3,3,64,128)),name='W14'), 
        'wc15': tf.Variable(tf.random.normal((3,3,64,64)),name='W15'), 
        'wc16': tf.Variable(tf.random.normal((3,3,64,64)),name='W16'), 
        'wc17': tf.Variable(tf.random.normal((3,3,32,64)),name='W17'), 
        'wc18': tf.Variable(tf.random.normal((3,3,32,32)),name='W18'), 
        'wc19': tf.Variable(tf.random.normal((3,3,32,32)),name='W19'), 
        'wc20': tf.Variable(tf.random.normal((3,3,32,16)),name='W20'), 
        'wc21': tf.Variable(tf.random.normal((3,3,16,3)),name='W21')
                }
        
        
        #self.placeholder_1 = tf.placeholder(tf.float32, shape=self.shape)
        
        #self.placeholder_2 = tf.placeholder(tf.float32, shape=self.shape)
        
        #self.placeholder_3 = tf.placeholder(tf.float32, shape=self.shape)
        
        #self.placeholder_4 = tf.placeholder(tf.float32, shape=self.shape)
        
        #self.placeholder_5 = tf.placeholder(tf.float32, shape=self.shape)
        
        #output_placeholder = tf.placeholder(tf.float32, shape =self.shape)
        self.trainable_variables=[self.lambda_1,self.lambda_2,self.lambda_3,self.lambda_4,self.lambda_5
                                 ,self.lambda_6,self.lambda_10,self.lambda_20,self.lambda_30,self.lambda_40,self.lambda_50
                                 ,self.lambda_60,self.weights_conv_encoder['wc1'],self.weights_conv_encoder['wc2'],self.weights_conv_encoder['wc3'],self.weights_conv_encoder['wc4']
                                 ,self.weights_conv_encoder['wc5'],self.weights_conv_encoder['wc6'],self.weights_conv_encoder['wc7'],self.weights_conv_decoder['wc1']
                                 ,self.weights_conv_decoder['wc2'],self.weights_conv_decoder['wc3'],self.weights_conv_decoder['wc4'],self.weights_conv_decoder['wc5'],self.weights_conv_decoder['wc6'],
                                 self.weights_conv_decoder['wc7'],self.weights_conv_decoder['wc8'],self.weights_cnn_final['wc1'],self.weights_cnn_final['wc2'],self.weights_cnn_final['wc3']
                                 ,self.weights_cnn_final['wc4'],self.weights_cnn_final['wc5'],self.weights_cnn_final['wc6'],self.weights_cnn_final['wc7'],self.weights_cnn_final['wc8'],self.weights_cnn_final['wc9']
                                 ,self.weights_cnn_final['wc10'] ]
    @tf.function    
    def MODEL(self,input_encoder,input_decoder,epochs):
        
        input_1=input_encoder[0]
        
        input_2=input_encoder[1]
        
        input_3=input_encoder[2]
        
        input_4=input_encoder[3]
        
        input_5=input_encoder[4]
        
        conv_1_output_encoder=self.conv_encoder(input_1)
        
        #return conv_1_output_encoder
        
            
        conv_2_output_encoder=self.conv_encoder(input_2)
        
        conv_3_output_encoder=self.conv_encoder(input_3)
        
        conv_4_output_encoder=self.conv_encoder(input_4)
        
        conv_5_output_encoder=self.conv_encoder(input_5)
        
        #return conv_5_output_encoder
 
        #final_loss= self.train_op(input_encoder,input_decoder,conv_5_output_encoder,epochs)
             
       
        
      
        #final_ans = self.cnn_final(final_decoder_concated)
        
        #print(final_ans)
        
        # final_loss = self.losses(final_ans,input_decoder[2])
        

        
        concated_encoder=self.concat(conv_1_output_encoder,conv_2_output_encoder,conv_3_output_encoder,conv_4_output_encoder,conv_5_output_encoder)
        #return concated_encoder

        encoder_output=self.autoencoder(concated_encoder)
        #return encoder_output
        
        #self.placeholder_decoder_1 = tf.placeholder(tf.float32, shape=self.shape)
        
        #self.placeholder_decoder_2 = tf.placeholder(tf.float32, shape=self.shape)
        
        input_decoder_1 = input_decoder[0]
        
        input_decoder_2 = input_decoder[1]
        
        conv1_decoder_output=self.conv_decoder(input_decoder_1)
        
        conv2_decoder_output=self.conv_decoder(input_decoder_2)
        
        concated_decoder_ref_1 = self.concat_decoder_ref(conv1_decoder_output,conv_1_output_encoder,input_decoder_1)
        
        concated_decoder_ref_2 = self.concat_decoder_ref(conv2_decoder_output,conv_2_output_encoder,input_decoder_2)
        
        
        
        conv_1_output_decoder22=self.conv_encoder22(concated_decoder_ref_1)
        
        conv_2_output_decoder22=self.conv_encoder22(concated_decoder_ref_2)
        
        conv_3_output_decoder21=self.conv_encoder21(conv_3_output_encoder)
        
        conv_4_output_decoder21=self.conv_encoder21(conv_4_output_encoder)
        
        conv_5_output_decoder21=self.conv_encoder21(conv_5_output_encoder)
        
        concated_decoder=self.concat(conv_1_output_decoder22,conv_2_output_decoder22,conv_3_output_decoder21,conv_4_output_decoder21,conv_5_output_decoder21)
        
        encoder_output_decoder=self.autoencoder_decoder(concated_decoder)
        
        concated_input_decoder =self.concat_decoder(input_decoder_1,input_decoder_2)
        
        output_decoder_12 = self.decoder_input_cnn(concated_input_decoder)
        
        #output_decoder_2 = self.decoder_input_cnn(input_decoder_2)
        
        final_decoder_concated = self.concat_decoder_final(encoder_output_decoder,encoder_output ,output_decoder_12)
        
        final_ans = self.cnn_final(final_decoder_concated)

        print(final_ans)
        #return final_ans
           # final_loss = self.losses(final_ans,input_decoder[2])
        
        final_loss= self.train_op(input_encoder,input_decoder,final_ans,epochs)
             
        return final_loss
            # functions used are conv_encoder , concat , autoencoder , conv_decoder , concat_decoder_ref , conv_encoder22 , conv_encoder21 , concat_decoder_final , autoencoder_final , autoencoder_decoder  
    
            # I have kept the weights of some layers same
         
    @tf.function
    def conv_encoder(self,x):
        
        x = tf.nn.conv2d(x, self.weights_conv_encoder['wc1'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_encoder['wc2'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_encoder['wc3'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_encoder['wc4'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_encoder['wc5'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_encoder['wc6'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_encoder['wc7'], strides=[1, 1,1, 1], padding='SAME')
        
        
        return x
    
    @tf.function    
    def concat(self,x1,x2,x3,x4,x5):
        
        x = tf.concat([x1,x2,x3,x4,x5],axis=-1)
        
        return x 
    
    @tf.function
    def conv_decoder(self,x):
        
        x = tf.nn.conv2d(x, self.weights_conv_decoder['wc1'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_decoder['wc2'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_decoder['wc3'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_decoder['wc4'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_decoder['wc5'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_decoder['wc6'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_decoder['wc7'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x, self.weights_conv_decoder['wc8'], strides=[1, 1,1, 1], padding='SAME')
        
        return x
    
    @tf.function
    def concat_decoder_ref(self,x1,x2,x3):
        
        x=tf.concat([x1,x2,x3],axis=-1)
        
        return x
    @tf.function
    def concat_decoder_final(self,x1,x2,x3):
        
        x=tf.concat([x1,x2,x3],axis=-1)
        
        return x
    
    @tf.function
    def autoencoder_decoder(self,x0):
        
        x = tf.nn.conv2d(x0,self.weights_autoencoder_decoder['wc1'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc19'] , strides=[1, 1,1, 1], padding='SAME')
        
        x1 = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc2'], strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.max_pool(x1, ksize=[1,3,3,1], strides=[1, 2 ,2, 1],padding='SAME') 
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc3'], strides=[1, 1,1, 1], padding='SAME')
        
        
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc4'] , strides=[1, 1,1, 1], padding='SAME')
        
        x2= tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc5'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.max_pool(x2, ksize=[1,3,3,1], strides=[1, 2 ,2, 1],padding='SAME') 
        
        x= tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc6'] , strides=[1, 1,1, 1], padding='SAME')
        
       
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc7'] , strides=[1, 1,1, 1], padding='SAME')
         
        x3 = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc8'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.max_pool(x3, ksize=[1,3,3,1], strides=[1, 2 ,2, 1],padding='SAME') 
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc9'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc10'] , strides=[1, 1,1, 1], padding='SAME')

        x = tf.nn.conv2d_transpose(x,self.weights_autoencoder_decoder['wc11'] , tf.constant([self.batch_size,self.calc1,self.calc2,128]) , strides=[1, 1,1, 1], padding='SAME')
        

        
        x=tf.math.add(tf.math.multiply(x,self.lambda_1),tf.math.multiply(self.lambda_2,x3))
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc12'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc13'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d_transpose(x,self.weights_autoencoder_decoder['wc14'] , tf.constant([self.batch_size,2*self.calc1,2*self.calc2,64]), strides=[1, 1,1, 1], padding='SAME')
        
        print(x.shape)
        
        x=tf.math.add(tf.math.multiply(x,self.lambda_3),tf.math.multiply(self.lambda_4,x2))
        
        print(x.shape)
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc15'] , strides=[1, 1,1, 1], padding='SAME')
        
        print(x.shape)
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc16'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d_transpose(x,self.weights_autoencoder_decoder['wc17'] ,tf.constant([self.batch_size,4*self.calc1,4*self.calc2,32]) , strides=[1, 1,1, 1], padding='SAME')
        

        
        x=tf.math.add(tf.math.multiply(x,self.lambda_5),tf.math.multiply(self.lambda_6,x1))
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc18'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc20'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x,self.weights_autoencoder_decoder['wc21'] , strides=[1, 1,1, 1], padding='SAME')
        
        
        
        
        #need to slice and uncomment




            #x = tf.add(x,tf.slice(x0))
        
        return x
    
    @tf.function   
    def autoencoder(self,x0):
    
        x = tf.nn.conv2d(x0,self.weights_autoencoder['wc1'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc19'] , strides=[1, 1,1, 1], padding='SAME')
        
        x1 = tf.nn.conv2d(x,self.weights_autoencoder['wc2'], strides=[1, 1,1, 1], padding='SAME')
        
        
        
        x = tf.nn.max_pool(x1, ksize=[1,3,3,1], strides=[1, 2 ,2, 1],padding='SAME') 
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc3'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc4'] , strides=[1, 1,1, 1], padding='SAME')
        
        x2 = tf.nn.conv2d(x,self.weights_autoencoder['wc5'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.max_pool(x2, ksize=[1,3,3,1], strides=[1, 2 ,2, 1],padding='SAME') 
    
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc6'] , strides=[1, 1,1, 1], padding='SAME')
         
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc7'] , strides=[1, 1,1, 1], padding='SAME')
        
        x3 = tf.nn.conv2d(x,self.weights_autoencoder['wc8'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.max_pool(x3, ksize=[1,3,3,1], strides=[1, 2 ,2, 1],padding='SAME') 
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc9'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc10'], strides=[1, 1,1, 1], padding='SAME')

        x = tf.nn.conv2d_transpose(x,self.weights_autoencoder['wc11'] , tf.constant([self.batch_size,self.calc1,self.calc2,128]) , strides=[1, 1,1, 1], padding='SAME')
        

        
        x=tf.math.add(tf.math.multiply(x,self.lambda_10),tf.math.multiply(self.lambda_20,x3))
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc12'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc13'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d_transpose(x,self.weights_autoencoder['wc14'] , tf.constant([self.batch_size,2*self.calc1,2*self.calc2,64]), strides=[1, 1,1, 1], padding='SAME')
        

        
        x=tf.math.add(tf.math.multiply(x,self.lambda_30),tf.math.multiply(self.lambda_40,x2))
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc15'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc16'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d_transpose(x,self.weights_autoencoder['wc17'] ,tf.constant([self.batch_size,4*self.calc1,4*self.calc2,32]) , strides=[1, 1,1, 1], padding='SAME')
        

        
        x=tf.math.add(tf.math.multiply(x,self.lambda_50),tf.math.multiply(self.lambda_60,x1))
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc18'] , strides=[1, 1,1, 1], padding='SAME')
        
        
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc20'] , strides=[1, 1,1, 1], padding='SAME')
        
        x = tf.nn.conv2d(x,self.weights_autoencoder['wc21'] , strides=[1, 1,1, 1], padding='SAME')
        
        
        
        
             #need to slice and uncomment





            #x = tf.add(x,tf.slice(x0))
        return x
    
            # need to see on tf.slice 
    
    @tf.function
    def conv_encoder22(self,x):
        
        x = tf.nn.conv2d(x, self.weights_conv_encoder22['wc1'], strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x, self.weights_conv_encoder22['wc2'], strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x, self.weights_conv_encoder22['wc3'], strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x, self.weights_conv_encoder22['wc4'], strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x, self.weights_conv_encoder22['wc5'], strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x, self.weights_conv_encoder22['wc6'], strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x, self.weights_conv_encoder22['wc7'], strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x, self.weights_conv_encoder22['wc8'], strides=[1, 1,1, 1], padding='SAME')
        
        return x
    @tf.function
    def conv_encoder21(self,x):    
        x = tf.nn.conv2d(x, self.weights_conv_encoder21['wc1'], strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x, self.weights_conv_encoder21['wc2'], strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x, self.weights_conv_encoder21['wc3'], strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x, self.weights_conv_encoder21['wc4'], strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x, self.weights_conv_encoder21['wc5'], strides=[1, 1,1, 1], padding='SAME')
    
        return x
    @tf.function
    def decoder_input_cnn(self,x):
        x = tf.nn.conv2d(x,self.weights_decoder_input_cnn['wc1'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_decoder_input_cnn['wc2'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_decoder_input_cnn['wc3'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_decoder_input_cnn['wc4'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_decoder_input_cnn['wc5'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_decoder_input_cnn['wc6'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_decoder_input_cnn['wc7'] , strides=[1, 1,1, 1], padding='SAME')
        
        return x
    tf.function
    def cnn_final(self,x):
        x = tf.nn.conv2d(x,self.weights_cnn_final['wc1'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_cnn_final['wc2'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_cnn_final['wc3'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_cnn_final['wc4'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_cnn_final['wc5'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_cnn_final['wc6'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_cnn_final['wc7'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_cnn_final['wc8'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_cnn_final['wc9'] , strides=[1, 1,1, 1], padding='SAME')
        x = tf.nn.conv2d(x,self.weights_cnn_final['wc10'] ,strides=[1, 1,1, 1], padding='SAME')
        
        return x
    #@tf.function
    def train(self,input_frames,target_frames,epoch_no):
        
        #_loss, _,output_frame = sess.run([self.final_loss , self.opt , self.final_ans], feed_dict={output_placeholder:target_frames[-1], 
        #self.placeholder_1 : input_frames[0], self.placeholder_2: input_frames[1], self.placeholder_3 : input_frames[2],
        #self.placeholder_4 : input_frames[3],self.placeholder_5 : input_frames[4], self.placeholder_decoder_1 : output_frames[0] ,
        #self.placeholder_decoder_2 : output_frames[1] })
                                                         
        output_frame =self.MODEL(input_frames,target_frames,epoch_no)
        #,loss,op
        
        
        final_loss= self.train_op(input_frames,target_frames,output_frame,epochs)
        #with tf.GradientTape() as tape:
            #output_frame,loss = self.Model(input_frames, target_frames,epoch_no.)
            #regularization_loss = tf.math.add_n(model.losses)
            #pred_loss = loss_fn(labels, predictions)
            #total_loss = pred_loss + regularization_loss

            #gradients = tape.gradient(loss, model.trainable_variables)
            #optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return output_frame
        #, loss,op
    
    @tf.function
    def losses(self,input_frame,target_frame):
        
        
            # once check on reduce_mean 
           # input_frame = tf.convert_to_tensor(input_frame, dtype=tf.float32)
            
            #target_frame = tf.convert_to_tensor(target_frame, dtype=tf.float32)
        
        #losse=tf.keras.losses.MSE(input_frame,target_frame)
        losse = tf.reduce_mean(tf.square(tf.math.subtract(input_frame , target_frame)))
        
        #tf.print(losse)
        
        return losse
    
    @tf.function
    def train_op(self,input_framess,output_framess,final_output,
                 epoch_no=1):
        
        #opti = tf.compat.v1.train.AdamOptimizer(self.lr_schedular(epoch_no))
        
        optimizer = tf.optimizers.Adam(self.lr_schedular(epoch_no))
        
        loss_value, grads = self.grad(input_framess[0],output_framess,final_output) 
        
        print(grads)
        print(loss_value)
        #optimizer.minimize(self.losses(final_output,output_framess[2]),var_list=self.trainable_variables)
        optimizer.apply_gradients(zip(grads,self.trainable_variables))
        
        return  loss_value
    @tf.function
    def grad(self,inputs,targets,final_output):
            
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                loss_value=self.losses(final_output, targets[2])
                
                print(loss_value)
                
                grads=tape.gradient(loss_value,self.trainable_variables)
            return loss_value, grads

            

            
        
            #train_op = opti.minimize(loss,var_list=[self.lambda_10],global_step=tf.compat.v1.train.get_or_create_global_step())
        
            
   
                           
    #@tf.function
    def lr_schedular(self,epoch_no = 1):
        
        if epoch_no<25 :
            
            return 1e-3
        elif epoch_no > 25 and epoch_no < 50:
            
            return 1e-4
        else:
            
            return 1e-5
    @tf.function()    
    def concat_decoder(self,x1,x2):
                
        x=tf.concat([x1,x2],axis=-1)
        
        return x
    
        
        
       
        
        

