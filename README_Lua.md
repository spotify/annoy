Install
-------

To install, you'll need Lua (binary + library) and LuaRocks.

If you have Python and Pip, you can get Lua and LuaRocks
using [hererocks](https://github.com/mpeterv/hererocks/),
written by Peter Melnichenko.

```
  pip install hererocks
  hererocks here --lua 5.1 --luarocks 2.2
```

This command installs Lua and LuaRocks locally to directory `here`.
To activate it, add `here/bin` to `PATH`:

```
  export PATH="$(pwd)/here/bin/:$PATH"
```

Then you can use commands `lua`, `luarocks`,
and tools installed by `luarocks`.

To build and install `annoy`, type:

```
  luarocks make
```

Background
----------

See the main README.

Lua code example
----------------

```lua
local annoy = require "annoy"

local f = 3
local t = annoy.AnnoyIndex(f) -- Length of item vector that will be indexed
for i = 0, 999 do
  local v = {math.random(), math.random(), math.random()}
  t:add_item(i, v)
end

t:build(10) -- 10 trees
t:save('test.ann')

-- ...

local u = annoy.AnnoyIndex(f)
u:load('test.ann') -- super fast, will just mmap the file

-- find the 10 nearest neighbors
local neighbors = u:get_nns_by_item(0, 10)
for rank, i in ipairs(neighbors) do
  print("neighbor", rank, "is", i)
end
```

Full Lua API
------------

Lua API closely resembles Python API, see main README.


Tests
-------

File `test/annoy_test.lua` is the literal translation of
`test/annoy_test.py` from Python+Nosetests to Lua+Busted.

To run tests, you need [Busted](http://olivinelabs.com/busted/),
Elegant Lua unit testing. To install it, type:

```
  luarocks install busted
```

To run tests, type:

```
  busted test/annoy_test.lua
```

It will take few minutes to execute.

Discuss
-------

There might be some memory leaks if inputs are incorrect.
Some functions allocate stack objects calling Lua functions throwing
Lua errors (e.g., `luaL_checkinteger`). A Lua error may omit calling
C++ destructors when unwinding the stack. (If it does, depends on
the Lua implementation and platform being in use.)

Lua binding was written by Boris Nagaev.
You can contact me via email (see https://github.com/starius).
