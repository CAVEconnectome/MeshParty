# MeshParty Development Guide for Claude

This document outlines common patterns, conventions, and important notes for developing with the MeshParty codebase.

## Code Patterns

### Circular Reference Prevention in __repr__
When implementing `__repr__` methods for classes that have circular references (e.g., parent-child relationships), avoid accessing child properties that might trigger infinite recursion. Instead:

```python
def __repr__(self) -> str:
    # Good: Use simple properties that don't trigger circular references
    layers = self.layers  # LayerManager.__repr__ returns simple list
    annos = self.annotations.names  # Direct property access
    return f"MeshWork(name={self.name}, layers={layers}, annotations={annos})"
    
    # Bad: Accessing child objects directly can cause recursion
    # return f"skel({self.skeleton.n_vertices})"  # May trigger skeleton.__repr__
```

### Iterator Implementation
Both `LayerManager` and `AnnotationManager` support iteration in insertion order:

```python
# Iterate over layer objects
for layer in mesh_work.layers:
    process_layer(layer)

# Iterate over annotation objects  
for annotation in mesh_work.annotations:
    process_annotation(annotation)
```

### Class Factory Pattern
Use `_from_existing` class methods for creating new instances from existing ones:

```python
@classmethod
def _from_existing(cls, new_morphsync, old_obj) -> Self:
    new_obj = cls(
        name=old_obj.name,
        morphsync=new_morphsync,
        meta=old_obj.meta,
    )
    # Copy layers and annotations using their own _from_existing methods
    for old_layer in old_obj.layers:
        new_layer = old_layer.__class__._from_existing(new_morphsync, old_layer)
        new_layer._register_meshworksync(new_obj)
        new_obj._layers.add(new_layer)
    return new_obj
```

## jj Conventions

### Method Documentation
Use clear, descriptive docstrings following NumPy style:

```python
def add_skeleton(
    self,
    vertices: Union[np.ndarray, pd.DataFrame, SkeletonSync],
    edges: Union[np.ndarray, pd.DataFrame],
    labels: Optional[Union[dict, pd.DataFrame]] = None,
    root: Optional[int] = None,
    *,
    vertex_index: Optional[Union[str, np.ndarray]] = None,
    linkage: Optional[Link] = None,
    spatial_columns: Optional[list] = None,
) -> Self:
    """
    Add a skeleton layer to the MorphSync.

    Parameters
    ----------
    vertices : Union[np.ndarray, pd.DataFrame, SkeletonSync]
        The vertices of the skeleton.
    edges : Union[np.ndarray, pd.DataFrame]
        The edges of the skeleton.
    labels : Optional[Union[dict, pd.DataFrame]]
        The labels for the skeleton.
    root : Optional[int]
        The root vertex for the skeleton, required if the edges are not already consistent with a single root.
    vertex_index : Optional[Union[str, np.ndarray]]
        The vertex index for the skeleton.
    linkage : Optional[Link]
        The linkage information for the skeleton. Typically, you will define the source vertices for the skeleton if using a graph-to-skeleton mapping.
    spatial_columns: Optional[list] = None
        The spatial columns for the skeleton, if vertices are a dataframe.

    Returns
    -------
    Self
    """
```

### Property Documentation
Document properties with brief, clear descriptions:

```python
@property
def labels(self) -> pd.DataFrame:
    """Return a DataFrame of all label columns across all layers."""
```

## Type Hints

### Use Union Types for Multiple Input Types
```python
vertices: Union[np.ndarray, pd.DataFrame, SkeletonSync]
labels: Optional[Union[dict, pd.DataFrame]] = None
```

### Self Return Type for Method Chaining
```python
def add_skeleton(self, ...) -> Self:
    # Enables method chaining
    return self
```

## Common Gotchas

1. **Circular References**: Be careful when accessing child objects in `__repr__` methods
2. **Property Access**: Some property accesses can trigger unexpected method calls in interactive environments
3. **Thread Safety**: The circular reference protection uses thread-local storage - be aware in multi-threaded contexts
4. **Method Chaining**: Many methods return `Self` to enable fluent interfaces

## Testing Patterns

When testing, check for:
- Proper handling of different input types (np.ndarray, pd.DataFrame, etc.)
- Method chaining works correctly
- `__repr__` doesn't cause infinite recursion
- Iterators return objects in expected order