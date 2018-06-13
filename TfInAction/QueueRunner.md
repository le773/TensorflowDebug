### 1.0 关于tensorflow 的数据读取线程管理QueueRunner

![AnimatedFileQueues.gif](https://i.imgur.com/OBSVmIP.gif)

### 2.0 string_input_producer
同时打开多个文件，显示创建Queue，同时隐含了QueueRunner的创建
#### 2.1 string_input_producer
```
def string_input_producer(string_tensor,
                          num_epochs=None,
                          shuffle=True,
                          seed=None,
                          capacity=32,
                          shared_name=None,
                          name=None,
                          cancel_op=None):
  """Output strings (e.g. filenames) to a queue for an input pipeline.
  Note: if `num_epochs` is not `None`, this function creates local counter
  `epochs`. Use `local_variables_initializer()` to initialize local variables.
  Args:
    string_tensor: A 1-D string tensor with the strings to produce.
    num_epochs: An integer (optional). If specified, `string_input_producer`
      produces each string from `string_tensor` `num_epochs` times before
      generating an `OutOfRange` error. If not specified,
      `string_input_producer` can cycle through the strings in `string_tensor`
      an unlimited number of times.
    shuffle: Boolean. If true, the strings are randomly shuffled within each
      epoch.
    seed: An integer (optional). Seed used if shuffle == True.
    capacity: An integer. Sets the queue capacity.
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions. All sessions open to the device which has
      this queue will be able to access it via the shared_name. Using this in
      a distributed setting means each name will only be seen by one of the
      sessions which has access to this operation.
    name: A name for the operations (optional).
    cancel_op: Cancel op for the queue (optional).
  Returns:
    A queue with the output strings.  A `QueueRunner` for the Queue
    is added to the current `Graph`'s `QUEUE_RUNNER` collection.
  Raises:
    ValueError: If the string_tensor is a null Python list.  At runtime,
    will fail with an assertion if string_tensor becomes a null tensor.
  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
  """
  not_null_err = "string_input_producer requires a non-null input tensor"
  if not isinstance(string_tensor, ops.Tensor) and not string_tensor:
    raise ValueError(not_null_err)

  with ops.name_scope(name, "input_producer", [string_tensor]) as name:
    string_tensor = ops.convert_to_tensor(string_tensor, dtype=dtypes.string)
    with ops.control_dependencies([
        control_flow_ops.Assert(
            math_ops.greater(array_ops.size(string_tensor), 0),
            [not_null_err])]):
      string_tensor = array_ops.identity(string_tensor)
    return input_producer(
        input_tensor=string_tensor,
        element_shape=[],
        num_epochs=num_epochs,
        shuffle=shuffle,
        seed=seed,
        capacity=capacity,
        shared_name=shared_name,
        name=name,
        summary_name="fraction_of_%d_full" % capacity,
        cancel_op=cancel_op)
```

#### 2.2 input_producer
```
def input_producer(input_tensor,
                   element_shape=None,
                   num_epochs=None,
                   shuffle=True,
                   seed=None,
                   capacity=32,
                   shared_name=None,
                   summary_name=None,
                   name=None,
                   cancel_op=None):
  """Output the rows of `input_tensor` to a queue for an input pipeline.
  Note: if `num_epochs` is not `None`, this function creates local counter
  `epochs`. Use `local_variables_initializer()` to initialize local variables.
  Args:
    input_tensor: A tensor with the rows to produce. Must be at least
      one-dimensional. Must either have a fully-defined shape, or
      `element_shape` must be defined.
    element_shape: (Optional.) A `TensorShape` representing the shape of a
      row of `input_tensor`, if it cannot be inferred.
    num_epochs: (Optional.) An integer. If specified `input_producer` produces
      each row of `input_tensor` `num_epochs` times before generating an
      `OutOfRange` error. If not specified, `input_producer` can cycle through
      the rows of `input_tensor` an unlimited number of times.
    shuffle: (Optional.) A boolean. If true, the rows are randomly shuffled
      within each epoch.
    seed: (Optional.) An integer. The seed to use if `shuffle` is true.
    capacity: (Optional.) The capacity of the queue to be used for buffering
      the input.
    shared_name: (Optional.) If set, this queue will be shared under the given
      name across multiple sessions.
    summary_name: (Optional.) If set, a scalar summary for the current queue
      size will be generated, using this name as part of the tag.
    name: (Optional.) A name for queue.
    cancel_op: (Optional.) Cancel op for the queue
  Returns:
    A queue with the output rows.  A `QueueRunner` for the queue is
    added to the current `QUEUE_RUNNER` collection of the current
    graph.
  Raises:
    ValueError: If the shape of the input cannot be inferred from the arguments.
    RuntimeError: If called with eager execution enabled.
  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
  """
  if context.executing_eagerly():
    raise RuntimeError(
        "Input pipelines based on Queues are not supported when eager execution"
        " is enabled. Please use tf.data to ingest data into your model"
        " instead.")
  with ops.name_scope(name, "input_producer", [input_tensor]):
    input_tensor = ops.convert_to_tensor(input_tensor, name="input_tensor")
    element_shape = input_tensor.shape[1:].merge_with(element_shape)
    if not element_shape.is_fully_defined():
      raise ValueError("Either `input_tensor` must have a fully defined shape "
                       "or `element_shape` must be specified")

    if shuffle:
      input_tensor = random_ops.random_shuffle(input_tensor, seed=seed)

    input_tensor = limit_epochs(input_tensor, num_epochs)

    q = data_flow_ops.FIFOQueue(capacity=capacity,
                                dtypes=[input_tensor.dtype.base_dtype],
                                shapes=[element_shape],
                                shared_name=shared_name, name=name)
    enq = q.enqueue_many([input_tensor])
    queue_runner.add_queue_runner(
        queue_runner.QueueRunner(
            q, [enq], cancel_op=cancel_op))
    if summary_name is not None:
      summary.scalar(summary_name,
                     math_ops.to_float(q.size()) * (1. / capacity))
    return q
```

### 3.0 使用案例
```
# Create the graph, etc.
init_op = tf.initialize_all_variables()

# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables (like the epoch counter).
sess.run(init_op)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while not coord.should_stop():
        # Run training steps or whatever
        sess.run(train_op)

except tf.errors.OutOfRangeError:
    print 'Done training -- epoch limit reached'
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
```