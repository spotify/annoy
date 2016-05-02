-- Copyright (c) 2016 Boris Nagaev
--
-- Licensed under the Apache License, Version 2.0 (the "License"); you may not
-- use this file except in compliance with the License. You may obtain a copy of
-- the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
-- WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
-- License for the specific language governing permissions and limitations under
-- the License.

package = "annoy"
version = "dev-1"
source = {
    url = "git://github.com/spotify/annoy.git",
}
description = {
    summary = "Approximate Nearest Neighbors Oh Yeah",
    homepage = "https://github.com/spotify/annoy",
    license = "Apache",
    detailed = [[
Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python
Go and Lua bindings to search for points in space that are close to a given
query point. It also creates large read-only file-based data structures
that are mmapped into memory so that many processes may share the same data.
]],
}
dependencies = {
    "lua >= 5.1",
}
build = {
    type = "builtin",
    modules = {
        ['annoy'] = {
            sources = {
                "src/annoyluamodule.cc",
            },
        },
    },
    platforms = {
        unix = {
            modules = {
                ['annoy'] = {
                    libraries = {"stdc++"},
                },
            },
        },
        mingw32 = {
            modules = {
                ['annoy'] = {
                    libraries = {"stdc++"},
                },
            },
        },
    },
}
