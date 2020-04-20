import tensorflow as tf

def custom_loss(ytrue,ypred):
    true_box_prob = ytrue[:,:2]
    true_box_coords1 = ytrue[:,2:6]
    true_box_coords2 = ytrue[:,6:10]
    pred_box_prob = ypred[:,:2]
    pred_box_coords1 = ypred[:,2:6]
    pred_box_coords2 = ypred[:,6:10]
    r1= tf.keras.losses.mse(y_true=true_box_coords1,y_pred=pred_box_coords1)
    r2= tf.keras.losses.mse(y_true=true_box_coords2,y_pred=pred_box_coords2)
    r1 = tf.multiply(r1 ,true_box_prob[:,0])
    r2 = tf.multiply(r2 ,true_box_prob[:,1])
    classification_loss = tf.keras.losses.binary_crossentropy(y_true=true_box_prob,y_pred=pred_box_prob)
    return (r1+r2)/1000.0  + classification_loss