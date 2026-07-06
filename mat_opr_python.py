from itertools import chain
import operator
try:
  import numpy as np
except ImportError:
  np = None


class array:
  '''It stores data as follows
  shape = (row_count, col_count)
  
  strides = (col_strides, row_strides)
  
  where col_stride tells how many
  elements to skip to get to the next
  element in a row'''
  
  def __init__(self, arr=None, shape=None, strides=None, r_limit=6, c_limit=6):
    if arr is None:
      arr = []
    self.r_limit = r_limit
    self.c_limit = c_limit
    
    if isinstance(arr, array):
      self.arr = arr.arr
      self.strides = arr.strides
      self.shape = arr.shape
    elif np is not None and isinstance(arr, np.ndarray):
      r, c = arr.shape if len(arr.shape)>1 else (1, arr.shape[0])
      
      byte = min(arr.strides)
      sr, sc = (arr.strides[1]//byte, arr.strides[0]//byte) if len(arr.strides)>1 else (1, c)
      
      self.shape = (r, c)
      self.strides = (sr, sc)
      self.arr = arr.ravel().tolist()
    else:
      if len(arr)==0:
        self.status = False
      else:
        
        if isinstance(arr[0], (int, float, bool)):
          self.status = False
        elif isinstance(arr[0], (array, list)):
          self.status = True
        else:
          
          raise ValueError("Provided datatype is not supported")
        
      if shape is None:
        if self.status:
          r = len(arr)
          c = len(arr[0])
          
        else:
          c = len(arr)
          r = 1 if len(arr) else 0
      else:
        r, c = shape
        
      self.shape=(r, c)
      if strides is not None:
        self.strides = strides
      else:
        self.strides = (1, c) if len(arr) else (0, 0)
      if self.status:
        
        self.arr = arr.arr if isinstance(arr, array) else list(chain.from_iterable(arr))
      else:
        self.arr = arr
      
  
  def copy(self):
    return array(self.arr.copy(), self.shape, self.strides)
  def array(self, arr):
    return array(arr)
  
  def __abs__(self):
    return array([abs(i) for i in self.arr], self.shape, self.strides)


  def _binary_op(self, other, op):
    # Applies an element-wise operation with broadcasting support
    
    if isinstance(other, (int, float)):
      return array([op(i, other) for i in self.arr], self.shape, self.strides)
    
    r, c = self.shape
    otr, otc = other.shape
    
    if (r!=otr and 1 not in (r, otr)) or (c!=otc and 1 not in (c, otc)):
      raise ValueError(f"Shape must be equal from left to right or one of them must be 1. Shape: {r, c} doesn't match with shape: {otr, otc}")
    nr, nc = max(r, otr), max(c, otc)
    asr, asc = self.strides
    bsr, bsc = other.strides
    
    '''
    Example:
    
    A = |1|      B = |1 2 3|
           |2|
    
    Broadcasting is performed row by row.
    
    If columns are broadcast, each element
    in a row is repeated `nc` times.
    
    If rows are broadcast, each row is
    repeated `nr` times.
    
    The code below implements
    broadcasting using this idea.
    Other implementations that produce the
    same behavior are valid.'''
    
    ar_state, ac_state = 1, 1
    br_state, bc_state = 1, 1
    if c<otc:
      ac_state = 0
    if r<otr:
      ar_state = 0
    if c>otc:
      bc_state = 0
    if r>otr:
      br_state = 0
    res = []
    for rr in range(nr):
      for cc in range(nc):
        a_ele = self.arr[rr*ar_state*asc+cc*ac_state*asr] # It sets the broadcasting axis index to be 0
        b_ele = other.arr[rr*br_state*bsc+cc*bc_state*bsr]# It sets the broadcasting axis index to be 0
        res.append(op(a_ele, b_ele))
    
    return array(res, (nr, nc), (1, nc))


  def __eq__(self, other): #==
    return self._binary_op(other, operator.eq)
  
  def __ne__(self, other): #!=
    return self._binary_op(other, operator.ne)
  
  def __lt__(self, other): #<
    return self._binary_op(other, operator.lt)
  
  def __le__(self, other): #<=
    return self._binary_op(other, operator.le)
  
  def __gt__(self, other): #>
    return self._binary_op(other, operator.gt)
  
  def __ge__(self, other): #>=
    return self._binary_op(other, operator.ge)
  
  def __radd__(self, other):
    # Allows broadcasting
    
    return self + other
  
  def __rmul__(self, other):
    # Allows broadcasting
    
    return self * other
  
  def __rsub__(self, other):
    # Allows broadcasting
    
    return array([other - i for i in self.arr], self.shape, self.strides)
  
  def __rtruediv__(self, other):
    # Returns a new array
    
    return array([other / i for i in self.arr], self.shape, self.strides)
  def __add__(self, other):
    return self._binary_op(other, operator.add)
  def __neg__(self):
    # Returns a new array
    
    return self * -1
  
  def __sub__(self, other):
    # Allows broadcasting
    # Returns a new array
    
    return self._binary_op(other, operator.sub)
  
  def __mul__(self, val):
    # Allows broadcasting
    # Returns a new array
    
    if isinstance(val, (int, float)):
      return array([i*val for i in self.arr], self.shape, self.strides)
    return self._binary_op(val, operator.mul)
  
  def __truediv__(self, other):
    # Allows broadcasting
    # Returns a new array
    
    return self._binary_op(other, operator.truediv)
  
  def __pow__(self, val):
    # Returns a new array of same shape but with new values
    return array([i**val for i in self.arr], self.shape, self.strides)
  
  def __iadd__(self, other):
    # Shapes must match
    # Changes the original values of self
    # No broadcasting allowed here
    
    tp = isinstance(other, (int, float, bool))
    for i in range(len(self)):
      self.arr[i] += other if tp else other.arr[i]
    return self
  
  def __isub__(self, other):
    # Shapes must match
    # Changes the original values of self
    # No broadcasting allowed here
    
    tp = isinstance(other, (int, float, bool))
    for i in range(len(self)):
      self.arr[i] -= other if tp else other.arr[i]
    return self
  
  def __imul__(self, other):
    # Shapes must match
    # Changes the original values of self
    # No broadcasting allowed here
    
    tp = isinstance(other, (int, float, bool))
    for i in range(len(self)):
      self.arr[i] *= other if tp else other.arr[i]
    return self
  
  def __itruediv__(self, other):
    # Shapes must match
    # Changes the original values of self
    # No broadcasting allowed here
    
    tp = isinstance(other, (int, float, bool))
    for i in range(len(self)):
      self.arr[i] /= other if tp else other.arr[i]
    return self
  
  
  def transpose(self):
    # Returns a view by swapping the shape and strides
    r, c = self.shape
    sr, sc = self.strides
    return array(self.arr, (c, r), (sc, sr))
  
  @property
  def T(self):
    return self.transpose()
  
  def reshape(self, r, c):
    # Returns a new array object sharing the same underlying data
    # with a different shape and strides.
    t = self.shape[0]*self.shape[1]
    
    if c<0 and r<0:
      raise ValueError(f"Cannot reshape array of size {self.shape} into {(r, c)}")
    if r == -1 and t%c==0:
      new_shape = (t//c, c)
    elif c == -1 and t%r==0:
      new_shape = (r, t//r)
    elif r*c == t:
      new_shape = (r, c)
    elif c*r != t:
      raise ValueError(f"Cannot reshape array of size {self.shape} into {(r, c)}")
    
    return array(self.arr, new_shape)
  
  def __matmul__(self, other):
    # Multiplies 2d matrices
    sr, sc = self.shape
    osr, osc = other.shape
    if sc!=osr:
      raise ValueError(f"Shape {sc} is different from {osr}")
    stdr, stdc = self.strides
    otdr, otdc = other.strides
    res = []
    
    row_pointer = 0
    for i in range(sr):
      col_pointer = 0
      for j in range(osc):
        rend = row_pointer+stdr*sc
        row_ind = range(row_pointer, rend, stdr)
        
        cend = col_pointer+otdc*osr
        col_ind = range(col_pointer, cend, otdc)
        
        res.append(sum(self.arr[i]*other.arr[j] for i, j in zip(row_ind, col_ind)))
        
        col_pointer += otdr # Skip 1 row stride
      row_pointer += stdc # Skip 1 column stride
    
    return array(res, (sr, osc))
  
  def __len__(self):
    return len(self.arr)
  

  def __getitem__(self, ind):
    '''Returns the item(s) based on the indices or slices given
    arr[m, n] --> single value at m`th row and n`th column
    arr[m] --> m`th row
    arr[:, n] --> n`th column
    arr[:, :] --> the entire array
    arr[m:n] --> row m to row n
    arr[:, m:n] --> column m to column n
    arr[mask] --> if mask shape and arr shape is same returns a row vector of values from arr where mask == True
    for example
    arr[arr>0] --> values where arr>0
    '''
    r, c = self.shape
    sr, sc = self.strides
    mask = False
    if isinstance(ind, (array, list)):
      arr_ind = array(ind.tolist())
      mask = True
      ind_sr, ind_sc = arr_ind.strides
      if self.shape != arr_ind.shape:
        raise ValueError(f"Shape must match. {self.shape} != {arr_ind.shape}")
      row_ind = range(r)
      col_ind = range(c)
    else:
      ln = 1 if not isinstance(ind, tuple) else len(ind)
      if ln>2:
        raise ValueError("Cannot handle dimension more than 2")
      if ln==1:
        row_ind = [ind] if not isinstance(ind, slice) else range(*ind.indices(r))
        if len(row_ind)==1 and row_ind[0]<0:
          row_ind[0] = r+ind
          if row_ind[0]<0:
            raise IndexError("Index out of range")
        col_ind = range(c)
      else:
        rp, cp = ind
        
        if isinstance(rp, int) and isinstance(cp, int):
          if rp>=r or cp>=c:
            raise IndexError("Index out of range")
          if rp<0:
            rp = r+rp
          if cp<0:
            cp = c+cp
          idx = sc*rp+sr*cp
          if idx<0:
            raise IndexError("Index out of range")
              
          return self.arr[idx]
        
        if isinstance(rp, int):
          if rp<0:
            rp = r+rp
            if rp<0:
              raise IndexError("Row index out of range")
          row_ind = [rp]
        else:
          row_ind = range(*rp.indices(r))
        
        if isinstance(cp, int):
          if cp<0:
            cp = c+cp
            if cp<0:
              raise IndexError("Column index out of range")
          col_ind = [cp]
        else:
          col_ind = range(*cp.indices(c))
        
    res = []
    
    for i in row_ind:
      for j in col_ind:
        if mask:
          if arr_ind.arr[ind_sr*j + ind_sc*i]:
            res.append(self.arr[i*sc+j*sr])
        else:
          res.append(self.arr[i*sc+j*sr])

    if mask:
      return array(res)
    return array(res, (len(row_ind), len(col_ind)))
  
  def __setitem__(self, mask, val):
    # Supports boolean mask and single index. Accepts a single value or a container as val
    if isinstance(mask, (list, array, tuple)) and len(mask)>0:
      if isinstance(mask.arr[0], bool) and len(mask) == len(self):
        
        is_mul = isinstance(val, (int, float, bool))
        for i, j in enumerate(mask.arr):
          if j:
            if isinstance(val, list):
              
              val_ = val[i]
             
            elif isinstance(val, array):
              
              val_ = val.arr[i]
              
            else:
              val_ = val
            self.arr[i] = val_
      else:
        raise ValueError("Must have same shape")
    elif isinstance(mask, int):
      self.arr[mask] = val
  
  def __sum__(self):
    return sum(self.arr)
  
  def mean(self, axis=None, **kwargs):
    #Calculates the mean along axis and from all elements if axis not provided
    if axis == None:
      return sum(self.arr)/len(self.arr)
    pointer = 0
    sr, sc = self.strides
    r, c = self.shape
    res = []
    """
      1 2 3
      4 5 6
      7 8 9"""
    if axis == -1:
      axis = 1
    if axis == -2:
      axis = 0
    if axis < -2 or axis>1:
      raise ValueError("Can only handle 2d array")
    if axis == 0:
      loop_run = c
      skip_ele = sr
      h_end = r
      sh_end = sc
      
    elif axis == 1:
      loop_run = r
      skip_ele = sc
      h_end = c
      sh_end = sr
        
    for i in range(loop_run):
      end = pointer + h_end*sh_end
      
      res.append(sum(self.arr[i] for i in range(pointer, end, sh_end))/h_end)
      
      pointer += skip_ele
        
    if axis == 0:
      new_shape = (1, c)
    else:
      new_shape = (r, 1)
    return array(res, new_shape)
  
  
  def std(self, axis=None, **kwargs):
    #Calculates the standard deviation along axis and from all elements if axis not provided
    if axis == None:
      diff = self - self.mean()
      return ((diff**2).mean())**.5
    diff = self - self.mean(axis=axis)
    return ((diff**2).mean(axis=axis))**.5
  
  def ravel(self):
    
    return self.reshape(1, -1)
  
  def flatten(self):
    return self.reshape(1, -1)
  
  def get_indices(self, size, limit):
      if size <= limit:
          return list(range(size))
  
      head = limit // 2
      tail = limit - head
  
      return (
          list(range(head))
          + [None]          # represents "..."
          + list(range(size - tail, size))
      )
  def tolist(self):
    # Returns a nested list.
    res = []
    pointer = 0
    r, c = self.shape
    
    sr, sc = self.strides
    for _ in range(r):
      tmp = []
      indices = range(pointer, pointer+c*sr, sr)
      for i in indices:
        tmp.append(self.arr[i])
      res.append(tmp)
      pointer += sc
    return res
  
  def any(self):
    return sum(self.arr)>0 # Returns True if at least one element is nonzero.

  def all(self):
    return self.arr.count(0)==0 # Returns True if no element is zero or False.

  
  def __invert__(self): # Magic method of ~
    return array([not i for i in self.arr], self.shape, self.strides)
  
  def __str__(self):
    r, c = self.shape
    sr, sc = self.strides
    rows = self.get_indices(r, self.r_limit)
    cols = self.get_indices(c, self.c_limit)
    
    res = "["
    
    for rr in rows:
        if rr is None:
            res += "...\n"
            continue
    
        line = []
    
        for cc in cols:
            if cc is None:
                line.append("...")
                continue
    
            idx = rr * sc + cc * sr
            ele = self.arr[idx]
            ele = f"{ele:.5f}" if isinstance(ele, float) else str(ele)
            line.append(ele)
    
        new_info = "[" + ", ".join(line) + "]\n"
        if len(new_info) > 3:
          res += new_info
    return res[:-1]+"]" if len(res)>1 else res+"]"
    


def where(mask, val1, val2):
  #Returns a new array replacing values where mask==True with val1 and val2 otherwise
  res = []
  is_mul1 = isinstance(val1, (int, float, bool))
  is_mul2 = isinstance(val2, (int, float, bool))
  
  for i, j in enumerate(mask.arr):
    pos = val1 if is_mul1 else val1.arr[i]
    neg = val2 if is_mul2 else val2.arr[i]
    if j:
      res.append(pos)
    else:
      res.append(neg)
  
  return array(res, mask.shape, mask.strides)



if __name__ == "__main__":
  
  #Compare the results agains numpy
  a = array([[1, 2, 3], [4, 5, 6]])
  b = array([[10], [20]])
  
  na = np.array([[1, 2, 3], [4, 5, 6]])
  nb = np.array([[10], [20]])
  
  print("Broadcasting")
  print("Custom:")
  print(a + b)
  print("NumPy:")
  print(na + nb)
  
  print("\nMatrix multiplication")
  print("Custom:")
  print(a.T @ a)
  print("NumPy:")
  print(na.T @ na)
  
  print("\nBoolean masking")
  mask = a > 3
  nmask = na > 3
  
  print("Custom mask:")
  print(mask)
  print("NumPy mask:")
  print(nmask)
  
  print("Custom selection:")
  print(a[mask])
  print("NumPy selection:")
  print(na[nmask])
  
  print("\nMean")
  print("Custom:")
  print(a.mean(axis=0))
  print("NumPy:")
  print(na.mean(axis=0))
  
  print("\nReshape")
  print("Custom:")
  print(a.reshape(3, 2))
  print("NumPy:")
  print(na.reshape(3, 2))