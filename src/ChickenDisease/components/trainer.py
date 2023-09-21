from src.ChickenDisease.constants import *
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ChickenDisease.config.configuration import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.trained_model_path
        )

    def data_split(self):
        df=pd.read_csv(self.config.training_data)
        trsplit=.9
        vsplit=.05
        dsplit=vsplit/(1-trsplit)

        strat=df['labels']
        self.train_df, dum_df= train_test_split(df, train_size= 0.9, shuffle=True, random_state=123, stratify=strat)

        strat=dum_df['labels']
        self.test_df,self.valid_df=train_test_split(dum_df,train_size=.9, shuffle=True, random_state=123, stratify=strat)
        print(f'train_df length:{len(self.train_df)}, test_df length:{len(self.test_df)},  valid_df length:{len(self.valid_df)}')

        
    
    def trim(self, max_size=500, min_size=0, column="labels"):
            df=self.train_df.copy() #copy the dataframe
            original_class_count= len(list(df['labels'].unique()))
            print("original class count is:", original_class_count)
            sample_list=[]
            groups= df.groupby('labels')

            for label in df[column].unique():
                group=groups.get_group(label)
                sample_count= len(group)
                if sample_count> max_size:
                    strat= group[column]
                    samples,_= train_test_split(group, train_size= max_size, shuffle=True, random_state=123, stratify= strat)
                    sample_list.append(samples)
                elif sample_count>=min_size:
                    sample_list.append(group)
            
            df=pd.concat(sample_list, axis=0).reset_index(drop=True)

            final_class_count=len(list(df[column].unique()))
            if final_class_count != original_class_count:
                print("***WARNING*** Data frame has been reduced")
            
            balance= list(df[column].value_counts())
            print(balance)
            return df


    def train_valid_generator(self):
        img_size= self.config.params_image_size
        batch_size= self.config.params_batch_size 
            
        trgen=tf.keras.preprocessing.image.ImageDataGenerator( rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest' )

        t_and_v_gen=tf.keras.preprocessing.image.ImageDataGenerator()

        self.train_gen=trgen.flow_from_dataframe(self.train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                        class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
        self.valid_gen=t_and_v_gen.flow_from_dataframe(self.valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                        class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
        # for the test_gen we want to calculate the batch size and test steps such that batch_size X test_steps= number of samples in test set
        # this insures that we go through all the sample in the test set exactly once.
        length=len(self.test_df)
        test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]  
        test_steps=int(length/test_batch_size)
        self.test_gen=t_and_v_gen.flow_from_dataframe(self.test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                        class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)
        # from the generator we can get information we will need later
        classes=list(self.train_gen.class_indices.keys())
        class_indices=list(self.train_gen.class_indices.values())
        class_count=len(classes)
        labels=self.test_gen.labels
        print ( 'test batch size: ' ,test_batch_size, '  test steps: ', test_steps, ' number of classes: ', class_count)
        print ('{0:^25s}{1:^12s}'.format('class name', 'class index'))
        for klass, index in zip(classes, class_indices):
            print(f'{klass:^25s}{str(index):^12s}')
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_gen.samples // self.train_gen.batch_size
        self.validation_steps = self.valid_gen.samples // self.valid_gen.batch_size

        self.model.fit(
            self.train_gen,
            epochs=self.config.params_epoch,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_gen,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
