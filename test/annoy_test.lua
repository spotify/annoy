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

local AnnoyIndex = require 'annoy'.AnnoyIndex

local function gauss(mu, sigma)
  local sum = -6
  for _ = 1, 12 do
    sum = sum + math.random()
  end
  return mu + sum * sigma
end

local function randomVector(f, mu, sigma)
  local v = {}
  for i = 1, f do
    v[i] = gauss(mu, sigma)
  end
  return v
end

local function round(x)
    return ("%.3f"):format(x)
end

local function roundArray(array)
    local rounded_array = {}
    for k, v in ipairs(array) do
        rounded_array[k] = round(v)
    end
    return rounded_array
end

local function isSorted(v)
    for i = 2, #v do
        if v[i-1] > v[i] then
            return false
        end
    end
    return true
end

local function max(array)
    local ans = assert(array[1])
    for _, v in ipairs(array) do
        ans = math.max(ans, v)
    end
    return ans
end

local function min(array)
    local ans = assert(array[1])
    for _, v in ipairs(array) do
        ans = math.min(ans, v)
    end
    return ans
end

local function precision(first1000, n, n_trees, n_points, n_rounds)
    if not n_trees then
        n_trees = 10
    end
    if not n_points then
        n_points = 10000
    end
    if not n_rounds then
        n_rounds = 10
    end
    local found = 0
    for _ = 1, n_rounds do
        local f = 10
        local p_size
        if first1000 then
            -- create random points at distance x from (1000, 0, 0, ...)
            p_size = f - 1
        else
            -- create random points at distance x
            p_size = f
        end
        local i = AnnoyIndex(f, 'euclidean')
        for j = 0, n_points - 1 do
            local p = randomVector(p_size, 0, 1)
            local norm
            do
                norm = 0
                for _, pi in ipairs(p) do
                    norm = norm + pi ^ 2
                end
                norm = norm ^ 0.5
            end
            local x = {}
            do
                if first1000 then
                    x[1] = 1000
                end
                for _, pi in ipairs(p) do
                    table.insert(x, pi / norm * j)
                end
            end
            i:add_item(j, x)
        end
        i:build(n_trees)
        local v = {}
        do
            for k = 1, f do
                v[k] = 0
            end
            if first1000 then
                v[1] = 1000
            end
        end
        local nns = i:get_nns_by_vector(v, n)
        assert(isSorted(nns))
        -- The number of gaps should be equal to the last item minus n-1
        for _, x in ipairs(nns) do
            if x < n then
                found = found + 1
            end
        end
    end
    return 1.0 * found / (n * n_rounds)
end

describe("angular annoy test", function()

    it("get_nns_by_vector", function()
        local f = 3
        local i = AnnoyIndex(f)
        i:add_item(0, {0, 0, 1})
        i:add_item(1, {0, 1, 0})
        i:add_item(2, {1, 0, 0})
        i:build(10)
        assert.same({2, 1, 0}, i:get_nns_by_vector({3, 2, 1}, 3))
        assert.same({0, 1, 2}, i:get_nns_by_vector({1, 2, 3}, 3))
        assert.same({2, 0, 1}, i:get_nns_by_vector({2, 0, 1}, 3))
    end)

    it("get_nns_by_item", function()
        local f = 3
        local i = AnnoyIndex(f)
        i:add_item(0, {2, 1, 0})
        i:add_item(1, {1, 2, 0})
        i:add_item(2, {0, 0, 1})
        i:build(10)
        assert.same({0, 1, 2}, i:get_nns_by_item(0, 3))
        assert.same({1, 0, 2}, i:get_nns_by_item(1, 3))
        do
            local close_to_2 = i:get_nns_by_item(2, 3)
            assert.equal(close_to_2[1], 2)
            assert.truthy(
                (close_to_2[2] == 0 and close_to_2[3] == 1)
                or
                (close_to_2[2] == 1 and close_to_2[3] == 0)
            )
        end
    end)

    it("dist", function()
        local f = 2
        local i = AnnoyIndex(f)
        i:add_item(0, {0, 1})
        i:add_item(1, {1, 1})
        assert.equal(round((2 * (1.0 - 2 ^ -0.5)) ^ 0.5), round(i:get_distance(0, 1)))
    end)

    it("dist_2", function()
        local f = 2
        local i = AnnoyIndex(f)
        i:add_item(0, {1000, 0})
        i:add_item(1, {10, 0})
        assert.equal(round(0), round(i:get_distance(0, 1)))
    end)

    it("dist_3", function()
        local f = 2
        local i = AnnoyIndex(f)
        i:add_item(0, {97, 0})
        i:add_item(1, {42, 42})
        local dist = ((1 - 2 ^ -0.5) ^ 2 + (2 ^ -0.5) ^ 2) ^ 0.5
        assert.equal(round(dist), round(i:get_distance(0, 1)))
    end)

    it("dist_degen", function()
        local f = 2
        local i = AnnoyIndex(f)
        i:add_item(0, {1, 0})
        i:add_item(1, {0, 0})
        assert.equal(round(2.0 ^ 0.5), round(i:get_distance(0, 1)))
    end)

    it("large_index", function()
        -- Generate pairs of random points where the pair is super close
        local f = 10
        local i = AnnoyIndex(f)
        for j = 0, 10000 - 1, 2 do
            local p = randomVector(f, 0, 1)
            local f1 = math.random() + 1
            local f2 = math.random() + 1
            local x = {}
            local y = {}
            for k, pi in ipairs(p) do
                x[k] = f1 * pi + gauss(0, 1e-2)
                y[k] = f2 * pi + gauss(0, 1e-2)
            end
            i:add_item(j, x)
            i:add_item(j+1, y)
        end
        i:build(10)
        for j = 0, 10000 - 1, 2 do
            assert.same({j, j+1}, i:get_nns_by_item(j, 2))
            assert.same({j+1, j}, i:get_nns_by_item(j+1, 2))
        end
    end)

    it("precision_1", function()
        assert.truthy(precision(true, 1) >= 0.98)
    end)

    it("precision_10", function()
        assert.truthy(precision(true, 10) >= 0.98)
    end)

    it("precision_100", function()
        assert.truthy(precision(true, 100) >= 0.98)
    end)

    it("precision_1000", function()
        assert.truthy(precision(true, 1000) >= 0.98)
    end)

    it("load_save_get_item_vector", function()
        local f = 3
        local i = AnnoyIndex(f)
        i:add_item(0, {1.1, 2.2, 3.3})
        i:add_item(1, {4.4, 5.5, 6.6})
        i:add_item(2, {7.7, 8.8, 9.9})
        assert.same(roundArray({1.1, 2.2, 3.3}), roundArray(i:get_item_vector(0)))
        assert.truthy(i:build(10))
        assert.truthy(i:save('blah.ann'))
        assert.same(roundArray({4.4, 5.5, 6.6}), roundArray(i:get_item_vector(1)))
        local j = AnnoyIndex(f)
        assert.truthy(j:load('blah.ann'))
        assert.same(roundArray({7.7, 8.8, 9.9}), roundArray(i:get_item_vector(2)))
    end)

    it("get_nns_search_k", function()
        local f = 3
        local i = AnnoyIndex(f)
        i:add_item(0, {0, 0, 1})
        i:add_item(1, {0, 1, 0})
        i:add_item(2, {1, 0, 0})
        i:build(10)
        assert.same({0, 1, 2}, i:get_nns_by_item(0, 3, 10))
        assert.same({2, 1, 0}, i:get_nns_by_vector({3, 2, 1}, 3, 10))
    end)

    it("include_dists", function()
        -- Double checking issue 112
        local f = 40
        local i = AnnoyIndex(f)
        local v = randomVector(f, 0, 1)
        i:add_item(0, v)
        local neg_v = {}
        do
            for k, value in ipairs(v) do
                neg_v[k] = -value
            end
        end
        i:add_item(1, neg_v)
        i:build(10)
        local indices, dists = i:get_nns_by_item(0, 2, 10, true)
        assert.same({0, 1}, indices)
        assert.same(roundArray({0.0, 2.0}), roundArray(dists))
    end)


    it("include_dists_check_ranges", function()
        local f = 3
        local i = AnnoyIndex(f)
        for j = 0, 100000 - 1 do
            i:add_item(j, randomVector(f, 0, 1))
        end
        i:build(10)
        local include_distances = true
        local _, dists = i:get_nns_by_item(0, 100000, -1, include_distances)
        assert.truthy(max(dists) < 2.0)
        assert.equal(round(0.0), round(min(dists)))
    end)

end)

describe("euclidean annoy test", function()

    it("get_nns_by_vector", function()
        local f = 2
        local i = AnnoyIndex(f, 'euclidean')
        i:add_item(0, {2, 2})
        i:add_item(1, {3, 2})
        i:add_item(2, {3, 3})
        i:build(10)
        assert.same({2, 1, 0}, i:get_nns_by_vector({4, 4}, 3))
        assert.same({0, 1, 2}, i:get_nns_by_vector({1, 1}, 3))
        assert.same({1, 2, 0}, i:get_nns_by_vector({4, 2}, 3))
    end)

    it("get_nns_by_item", function()
        local f = 2
        local i = AnnoyIndex(f, 'euclidean')
        i:add_item(0, {2, 2})
        i:add_item(1, {3, 2})
        i:add_item(2, {3, 3})
        i:build(10)
        assert.same({0, 1, 2}, i:get_nns_by_item(0, 3))
        assert.same({2, 1, 0}, i:get_nns_by_item(2, 3))
    end)

    it("dist", function()
        local f = 2
        local i = AnnoyIndex(f, 'euclidean')
        i:add_item(0, {0, 1})
        i:add_item(1, {1, 1})
        assert.equal(round(1.0), round(i:get_distance(0, 1)))
    end)

    it("large_index", function()
        -- Generate pairs of random points where the pair is super close
        local f = 10
        -- local q = randomVector(f, 0, 10)
        local i = AnnoyIndex(f, 'euclidean')
        for j = 0, 10000 - 1, 2 do
            local p = randomVector(f, 0, 1)
            local x = {}
            local y = {}
            for k, pi in ipairs(p) do
                x[k] = 1 + pi + gauss(0, 1e-2) -- todo: should be q[i]
                y[k] = 1 + pi + gauss(0, 1e-2)
            end
            i:add_item(j, x)
            i:add_item(j+1, y)
        end
        i:build(10)
        for j = 0, 10000 - 1, 2 do
            assert.same({j, j+1}, i:get_nns_by_item(j, 2))
            assert.same({j+1, j}, i:get_nns_by_item(j+1, 2))
        end
    end)

    it("precision_1", function()
        assert.truthy(precision(false, 1) >= 0.98)
    end)

    it("precision_10", function()
        assert.truthy(precision(false, 10) >= 0.98)
    end)

    it("precision_100", function()
        assert.truthy(precision(false, 100) >= 0.98)
    end)

    it("precision_1000", function()
        assert.truthy(precision(false, 1000) >= 0.98)
    end)

    it("get_nns_with_distances", function()
        local f = 3
        local i = AnnoyIndex(f, 'euclidean')
        i:add_item(0, {0, 0, 2})
        i:add_item(1, {0, 1, 1})
        i:add_item(2, {1, 0, 0})
        i:build(10)
        do
            local l, d = i:get_nns_by_item(0, 3, -1, true)
            assert.same({0, 1, 2}, l)
            assert.same(
                roundArray({0, 2, 5}),
                roundArray({d[1]^2, d[2]^2, d[3]^2})
            )
        end
        do
            local l, d = i:get_nns_by_vector({2, 2, 2}, 3, -1, true)
            assert.same({1, 0, 2}, l)
            assert.same(
                roundArray({6, 8, 9}),
                roundArray({d[1]^2, d[2]^2, d[3]^2})
            )
        end
    end)

    it("include_dists", function()
        local f = 40
        local i = AnnoyIndex(f)
        local v = randomVector(f, 0, 1)
        i:add_item(0, v)
        local neg_v = {}
        do
            for k, value in ipairs(v) do
                neg_v[k] = -value
            end
        end
        i:add_item(1, neg_v)
        i:build(10)
        local indices, dists = i:get_nns_by_item(0, 2, 10, true)
        assert.same({0, 1}, indices)
        assert.same(round(0.0), round(dists[1]))
    end)

end)

describe("index test", function()

    it("not_found_tree", function()
        local i = AnnoyIndex(10)
        assert.has_error(function()
            i:load('nonexists.tree')
        end)
    end)

    it("binary_compatibility", function()
        local i = AnnoyIndex(10)
        i:load('test/test.tree')

        -- This might change in the future if we change the search
        -- algorithm, but in that case let's update the test
        assert.same(
            {0, 85, 42, 11, 54, 38, 53, 66, 19, 31},
            i:get_nns_by_item(0, 10)
        )
    end)

    it("load_unload", function()
        -- Issue #108
        local i = AnnoyIndex(10)
        for _ = 1, 100000 do
            i:load('test/test.tree')
            i:unload()
        end
    end)

    it("construct_load_destruct", function()
        for x = 1, 100000 do
            local i = AnnoyIndex(10)
            i:load('test/test.tree')
            if x % 100 == 0 then
                collectgarbage()
            end
        end
    end)

    it("construct_destruct", function()
        for _ = 1, 100000 do
            local i = AnnoyIndex(10)
            i:add_item(1000, randomVector(10, 0, 1))
        end
    end)

    it("save_twice", function()
        -- Issue #100
        local t = AnnoyIndex(10)
        t:save("t.ann")
        t:save("t.ann")
    end)

    it("load_save", function()
        -- Issue #61
        local i = AnnoyIndex(10)
        i:load('test/test.tree')
        local u = i:get_item_vector(99)
        i:save('x.tree')
        local v = i:get_item_vector(99)
        assert.same(u, v)
        local j = AnnoyIndex(10)
        j:load('test/test.tree')
        local w = i:get_item_vector(99) -- maybe s/i/j/?
        assert.same(u, w)
    end)

    it("save_without_build", function()
        -- Issue #61
        local i = AnnoyIndex(10)
        i:add_item(1000, randomVector(10, 0, 1))
        i:save('x.tree')
        local j = AnnoyIndex(10)
        j:load('x.tree')
        j:build(10)
    end)
end)

describe("types test", function()

    local n_points = 1000
    local n_trees = 10

    -- tests "numpy" and "tuple" are not applicable to Lua

    it("wrong_length", function()
        local f = 10
        local i = AnnoyIndex(f, 'euclidean')
        i:add_item(0, randomVector(f, 0, 1))
        assert.has_error(function()
            i:add_item(1, randomVector(f + 1000, 0, 1))
        end)
        assert.has_error(function()
            i:add_item(2, {})
        end)
        i:build(n_trees)
    end)

    it("range_errors", function()
        local f = 10
        local i = AnnoyIndex(f, 'euclidean')
        for j = 0, n_points - 1 do
            i:add_item(j, randomVector(f, 0, 1))
        end
        assert.has_error(function()
            i:add_item(-1, randomVector(f))
        end)
        i:build(n_trees)
        for _, bad_index in ipairs({-1000, -1, n_points, n_points + 1000}) do
            assert.has_error(function()
                i:get_distance(0, bad_index)
            end)
            assert.has_error(function()
                i:get_nns_by_item(bad_index, 1)
            end)
            assert.has_error(function()
                i:get_item_vector(bad_index)
            end)
        end
    end)

end)

describe("memory leaks", function()

    it("get_item_vector", function()
        local f = 10
        local i = AnnoyIndex(f, 'euclidean')
        i:add_item(0, randomVector(f, 0, 1))
        for j = 0, 100 - 1 do
            print(j, '...')
            for _ = 1, 1000 * 1000 do
                i:get_item_vector(0)
            end
        end
    end)

    it("get_lots_of_nns", function()
        local f = 10
        local i = AnnoyIndex(f, 'euclidean')
        i:add_item(0, randomVector(f, 0, 1))
        i:build(10)
        for _ = 1, 100 do
            assert.same({0}, i:get_nns_by_item(0, 999999999))
        end
    end)

end)
