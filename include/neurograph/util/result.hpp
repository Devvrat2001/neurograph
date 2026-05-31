#pragma once

#include <variant>
#include <string>
#include <optional>
#include <stdexcept>

namespace neurograph::util {

struct Error {
    std::string message;
    int code = 0;

    Error() = default;
    explicit Error(std::string msg, int c = 0)
        : message(std::move(msg)), code(c) {}
};

/// Lightweight Result type for explicit error handling without exceptions.
/// Holds either a value of type T or an Error.
template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(Error error) : data_(std::move(error)) {}

    bool ok() const noexcept { return std::holds_alternative<T>(data_); }
    bool is_error() const noexcept { return std::holds_alternative<Error>(data_); }

    const T& value() const& {
        if (is_error()) throw std::runtime_error(error().message);
        return std::get<T>(data_);
    }

    T& value() & {
        if (is_error()) throw std::runtime_error(error().message);
        return std::get<T>(data_);
    }

    T value() && {
        if (is_error()) throw std::runtime_error(error().message);
        return std::get<T>(std::move(data_));
    }

    const Error& error() const& {
        return std::get<Error>(data_);
    }

    explicit operator bool() const noexcept { return ok(); }

private:
    std::variant<T, Error> data_;
};

/// Specialization for void results (success/failure with no value).
template <>
class Result<void> {
public:
    Result() : error_(std::nullopt) {}
    Result(Error error) : error_(std::move(error)) {}

    bool ok() const noexcept { return !error_.has_value(); }
    bool is_error() const noexcept { return error_.has_value(); }

    const Error& error() const& {
        return error_.value();
    }

    explicit operator bool() const noexcept { return ok(); }

    static Result success() { return Result(); }

private:
    std::optional<Error> error_;
};

} // namespace neurograph::util
