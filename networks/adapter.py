1.backbone的灵活配置
2.adapter的灵活配置
3.



def clip_gradients(params, grad_and_vars):
    # clip gradient to prevent inf loss
    if params.max_grad_norm > 0:
        clipped_grads_and_vars = []
        for grad, var in grad_and_vars:
            if grad is not None:
                if isinstance(grad, tf.IndexedSlices):
                    tmp = tf.clip_by_norm(grad.values, params.max_grad_norm)
                    grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
                else:
                    grad = tf.clip_by_norm(grad, params.max_grad_norm)
            clipped_grads_and_vars.append((grad, var))
        return clipped_grads_and_vars
    return grad_and_vars


   def restore(self, restore_path):
        """Restore Name Match Op
        Args:
            restore_path: path to the restore file(no need to specify the suffix(.meta etc))
        """
        reader = tf.train.NewCheckpointReader(restore_path)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in self.model.variables
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0], self.model.variables), self.model.variables))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    tf.logging.info("Restored var %s" % var_name)
                    restore_vars.append(curr_var)
                else:
                    tf.logging.info("Wrong var shape %s" % var_name)
                    # import pdb;pdb.set_trace()
        saver = tfe.Saver(restore_vars)
        saver.restore(restore_path)
        if self.params.restore_from_ground == False:
            best_acc = restore_path.split('/')[-1].split('_')[-3]
            self.best_eval_acc = float(best_acc)
            tf.logging.info('Restored best acc" %.4f' % (self.best_eval_acc))