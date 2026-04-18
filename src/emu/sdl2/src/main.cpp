/*
 * Copyright (c) 2024 EKA2L1 Team.
 *
 * This file is part of EKA2L1 project.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <sdl2/emu_window_sdl2.h>
#include <sdl2/gl_context_sdl2.h>

#include <common/algorithm.h>
#include <common/arghandler.h>
#include <common/buffer.h>
#include <common/cvt.h>
#include <common/fileutils.h>
#include <common/log.h>
#include <common/path.h>
#include <common/pystr.h>
#include <common/sync.h>
#include <common/thread.h>
#include <common/version.h>

#include <config/app_settings.h>
#include <config/config.h>

#include <drivers/audio/audio.h>
#include <drivers/graphics/graphics.h>
#include <drivers/input/common.h>
#include <drivers/sensor/sensor.h>

#include <kernel/kernel.h>
#include <kernel/libmanager.h>

#include <loader/mbm.h>
#include <loader/mif.h>
#include <loader/svgb.h>
#include <loader/nvg.h>
#include <lunasvg.h>
#include <package/manager.h>
#include <services/applist/applist.h>
#include <services/fbs/fbs.h>
#include <services/fbs/bitmap.h>
#include <services/init.h>
#include <services/window/screen.h>
#include <services/window/window.h>

#include <drivers/itc.h>
#include <drivers/input/common.h>

#include <system/devices.h>
#include <system/epoc.h>
#include <system/installation/rpkg.h>

#include <utils/apacmd.h>
#include <vfs/vfs.h>

#include <gdbstub/gdbstub.h>

#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

namespace eka2l1::sdl {
    static std::atomic<bool> process_termination_requested{ false };

    static void handle_process_termination_signal(int) {
        process_termination_requested.store(true);
    }

    struct emulator_state {
        std::unique_ptr<system> symsys;
        std::unique_ptr<drivers::graphics_driver> graphics_driver;
        std::unique_ptr<drivers::audio_driver> audio_driver;
        std::unique_ptr<drivers::sensor_driver> sensor_driver;
        std::unique_ptr<config::app_settings> app_settings;
        std::unique_ptr<emu_window_sdl2> window;

        std::atomic<bool> should_emu_quit{false};
        std::atomic<bool> should_emu_pause{false};
        std::atomic<bool> stage_two_inited{false};
        std::atomic<bool> app_exited{false};

        bool app_launch_from_command_line = false;

        common::event graphics_event;
        common::event init_event;
        common::event pause_event;
        common::event kill_event;

        config::state conf;
        window_server *winserv = nullptr;
        std::mutex lockdown;
        std::size_t sys_reset_cbh = 0;

        int present_status = 0;
        std::atomic<int> host_rotation{0};
        std::atomic<bool> show_osd_requested{false};

        std::atomic<bool> osd_visible{false};
        std::atomic<bool> app_started{false};
        std::mutex osd_mutex;
        std::vector<uint8_t> osd_pixels;
        int osd_w = 0, osd_h = 0;
        bool osd_texture_dirty = false;
        drivers::handle osd_texture = 0;

        void stage_one();
        bool stage_two();
        void on_system_reset(system *the_sys);
        void register_draw_callback();
    };

    void emulator_state::stage_one() {
        log::setup_log(nullptr);
        log::toggle_console();

        conf.deserialize();
        if (log::filterings) {
            log::filterings->parse_filter_string(conf.log_filter);
        }

        if (conf.keybinds.keybinds.empty()) {
            auto add_key = [this](std::uint32_t sdl_keycode, std::uint32_t target_scancode) {
                config::keybind kb;
                kb.source.type = config::KEYBIND_TYPE_KEY;
                kb.source.data.keycode = sdl_keycode;
                kb.target = target_scancode;
                conf.keybinds.keybinds.push_back(kb);
            };

            add_key(SDLK_UP, epoc::std_key_up_arrow);
            add_key(SDLK_DOWN, epoc::std_key_down_arrow);
            add_key(SDLK_LEFT, epoc::std_key_left_arrow);
            add_key(SDLK_RIGHT, epoc::std_key_right_arrow);
            add_key(SDLK_RETURN, epoc::std_key_device_3);     // Enter = OK/Select
            add_key(SDLK_F1, epoc::std_key_device_0);         // F1 = Left Softkey
            add_key(SDLK_F2, epoc::std_key_device_1);         // F2 = Right Softkey
            add_key(SDLK_F3, epoc::std_key_application_0);    // F3 = Green/Call
            add_key(SDLK_F4, epoc::std_key_application_1);    // F4 = Red/End
            add_key(SDLK_BACKSPACE, epoc::std_key_backspace);
            add_key(SDLK_SPACE, epoc::std_key_space);
            add_key(SDLK_0, '0');
            add_key(SDLK_1, '1');
            add_key(SDLK_2, '2');
            add_key(SDLK_3, '3');
            add_key(SDLK_4, '4');
            add_key(SDLK_5, '5');
            add_key(SDLK_6, '6');
            add_key(SDLK_7, '7');
            add_key(SDLK_8, '8');
            add_key(SDLK_9, '9');
            add_key(SDLK_ESCAPE, epoc::std_key_no);           // Esc = Red/End key

            LOG_INFO(FRONTEND_CMDLINE, "Default keyboard bindings created ({} bindings)", conf.keybinds.keybinds.size());
        }

        LOG_INFO(FRONTEND_CMDLINE, "EKA2L1 SDL2 frontend v0.0.1 ({}-{})", GIT_BRANCH, GIT_COMMIT_HASH);
        app_settings = std::make_unique<config::app_settings>(&conf);

        system_create_components comp;
        comp.audio_ = nullptr;
        comp.graphics_ = nullptr;
        comp.conf_ = &conf;
        comp.settings_ = app_settings.get();

        symsys = std::make_unique<eka2l1::system>(comp);

        device_manager *dvcmngr = symsys->get_device_manager();

        if (dvcmngr->total() > 0) {
            symsys->startup();

            if (conf.enable_gdbstub) {
                symsys->get_gdb_stub()->set_server_port(conf.gdb_port);
            }

            if (!symsys->set_device(conf.device)) {
                LOG_ERROR(FRONTEND_CMDLINE, "Failed to set device index {}, falling back to 0", conf.device);
                conf.device = 0;
                symsys->set_device(0);
            }

            symsys->mount(drive_c, drive_media::physical, add_path(conf.storage, "/drives/c/"), io_attrib_internal);
            symsys->mount(drive_d, drive_media::physical, add_path(conf.storage, "/drives/d/"), io_attrib_internal);
            symsys->mount(drive_e, drive_media::physical, add_path(conf.storage, "/drives/e/"), io_attrib_removeable);

            on_system_reset(symsys.get());
        }

        sys_reset_cbh = symsys->add_system_reset_callback([this](system *the_sys) {
            on_system_reset(the_sys);
        });

        stage_two_inited = false;
    }

    bool emulator_state::stage_two() {
        if (!stage_two_inited) {
            device_manager *dvcmngr = symsys->get_device_manager();
            device *dvc = dvcmngr->get_current();

            if (!dvc) {
                LOG_ERROR(FRONTEND_CMDLINE, "No current device available. Stage two aborted.");
                return false;
            }

            LOG_INFO(FRONTEND_CMDLINE, "Device: {} ({})", dvc->model, dvc->firmware_code);

            symsys->mount(drive_z, drive_media::rom,
                add_path(conf.storage, "/drives/z/"), io_attrib_internal | io_attrib_write_protected);

            drivers::player_type player_be = drivers::player_type_tsf;
            if (conf.midi_backend == config::MIDI_BACKEND_MINIBAE)
                player_be = drivers::player_type_minibae;

            audio_driver = drivers::make_audio_driver(drivers::audio_driver_backend::cubeb,
                conf.audio_master_volume, player_be);

            if (audio_driver) {
                audio_driver->set_bank_path(drivers::MIDI_BANK_TYPE_HSB, conf.hsb_bank_path);
                audio_driver->set_bank_path(drivers::MIDI_BANK_TYPE_SF2, conf.sf2_bank_path);
            }

            symsys->set_audio_driver(audio_driver.get());

            sensor_driver = drivers::sensor_driver::instantiate();
            symsys->set_sensor_driver(sensor_driver.get());
            symsys->initialize_user_parties();

            if (!conf.svg_icon_cache_reset) {
                common::delete_folder("cache\\");
                conf.svg_icon_cache_reset = true;
                conf.serialize(false);
            }

            std::vector<std::tuple<std::u16string, std::string, epocver>> dlls_need_to_copy = {
                { u"Z:\\sys\\bin\\goommonitor.dll", "patch\\goommonitor_general.dll", epocver::epoc94 },
                { u"Z:\\sys\\bin\\avkonfep.dll", "patch\\avkonfep_general.dll", epocver::epoc93fp1 }
            };

            io_system *io = symsys->get_io_system();

            for (auto &[org_path, patch_path, ver_required] : dlls_need_to_copy) {
                if (symsys->get_symbian_version_use() < ver_required)
                    continue;

                auto where_to_copy = io->get_raw_path(org_path);
                if (where_to_copy.has_value()) {
                    std::string dest = common::ucs2_to_utf8(where_to_copy.value());
                    std::string backup = dest + ".bak";
                    if (common::exists(dest) && !common::exists(backup))
                        common::move_file(dest, backup);
                    common::copy_file(patch_path, dest, true);
                }
            }

            manager::packages *pkgmngr = symsys->get_packages();
            pkgmngr->load_registries();
            pkgmngr->migrate_legacy_registries();

            stage_two_inited = true;
        }

        return true;
    }

    void emulator_state::on_system_reset(system *the_sys) {
        winserv = reinterpret_cast<window_server *>(the_sys->get_kernel_system()->get_by_name<service::server>(
            get_winserv_name_by_epocver(symsys->get_symbian_version_use())));

        if (winserv) {
            winserv->on_all_clients_disconnected = [this]() {
                if (app_started.load())
                    std::exit(0);
            };
        }

        if (stage_two_inited) {
            register_draw_callback();
            symsys->initialize_user_parties();
        }
    }

    static void draw_screen_impl(emulator_state *state, epoc::screen *scr, const bool is_dsa);

    static void draw_screen_callback(void *userdata, epoc::screen *scr, const bool is_dsa) {
        emulator_state *state = reinterpret_cast<emulator_state *>(userdata);
        if (!state || !state->graphics_driver) {
            return;
        }

        try {
            draw_screen_impl(state, scr, is_dsa);
        } catch (const std::exception &e) {
            LOG_ERROR(FRONTEND_CMDLINE, "draw_screen_callback exception: {}", e.what());
        }
    }

    static void draw_screen_impl(emulator_state *state, epoc::screen *scr, const bool is_dsa) {
        state->app_started.store(true);
        // Register disconnect callback here in case winserv was null during on_system_reset
        // (which happens in --app mode where stage_two() hasn't run yet at reset time)
        if (state->winserv && !state->winserv->on_all_clients_disconnected) {
            state->winserv->on_all_clients_disconnected = [state]() {
                if (state->app_started.load())
                    std::exit(0);
            };
        }
        state->graphics_driver->wait_for(&state->present_status);

        const int total_rotation = (scr->ui_rotation + state->host_rotation.load()) % 360;

        drivers::graphics_command_builder builder;

        const auto window_width = state->window->window_fb_size().x;
        const auto window_height = state->window->window_fb_size().y;

        eka2l1::vec2 swapchain_size(window_width, window_height);
        builder.set_swapchain_size(swapchain_size);
        builder.backup_state();

        builder.set_feature(drivers::graphics_feature::cull, false);
        builder.set_feature(drivers::graphics_feature::depth_test, false);
        builder.set_feature(drivers::graphics_feature::blend, false);
        builder.set_feature(drivers::graphics_feature::stencil_test, false);
        builder.set_feature(drivers::graphics_feature::clipping, false);

        eka2l1::rect viewport;
        viewport.size = swapchain_size;
        builder.set_viewport(viewport);

        builder.clear({ 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f }, drivers::draw_buffer_bit_color_buffer);

        auto &crr_mode = scr->current_mode();
        eka2l1::vec2 size = crr_mode.size;
        if ((total_rotation % 180) != 0) {
            std::swap(size.x, size.y);
        }

        float mult_x = static_cast<float>(window_width) / size.x;
        float mult_y = mult_x;
        float width = size.x * mult_x;
        float height = size.y * mult_y;

        if (height > swapchain_size.y) {
            height = swapchain_size.y;
            mult_x = mult_y = height / size.y;
            width = size.x * mult_y;
        }

        std::uint32_t x = (swapchain_size.x - static_cast<std::uint32_t>(width)) / 2;
        std::uint32_t y = (swapchain_size.y - static_cast<std::uint32_t>(height)) / 2;

        scr->set_native_scale_factor(state->graphics_driver.get(), mult_x, mult_y);
        scr->absolute_pos.x = static_cast<int>(x);
        scr->absolute_pos.y = static_cast<int>(y);

        eka2l1::rect dest;
        dest.top = eka2l1::vec2(x, y);
        dest.size = eka2l1::vec2(static_cast<int>(width), static_cast<int>(height));

        eka2l1::rect src;
        src.size = crr_mode.size;
        src.size *= scr->display_scale_factor;

        drivers::advance_draw_pos_around_origin(dest, total_rotation);
        if (total_rotation % 180 != 0) {
            std::swap(dest.size.x, dest.size.y);
        }

        builder.set_texture_filter(scr->screen_texture, true, drivers::filter_option::linear);
        builder.set_texture_filter(scr->screen_texture, false, drivers::filter_option::linear);

        builder.draw_bitmap(scr->screen_texture, 0, dest, src, eka2l1::vec2(0, 0),
            static_cast<float>(total_rotation), 0);

        if (state->osd_visible.load() && state->osd_texture != 0) {
            builder.set_feature(drivers::graphics_feature::blend, true);
            builder.blend_formula(
                drivers::blend_equation::add, drivers::blend_equation::add,
                drivers::blend_factor::frag_out_alpha, drivers::blend_factor::one_minus_frag_out_alpha,
                drivers::blend_factor::one, drivers::blend_factor::one_minus_frag_out_alpha);

            {
                const std::lock_guard<std::mutex> guard(state->osd_mutex);
                if (state->osd_texture_dirty) {
                    builder.update_bitmap(state->osd_texture,
                        reinterpret_cast<const char *>(state->osd_pixels.data()),
                        state->osd_pixels.size(),
                        eka2l1::vec2(0, 0),
                        eka2l1::vec2(state->osd_w, state->osd_h));
                    state->osd_texture_dirty = false;
                }
            }

            eka2l1::rect osd_dest;
            osd_dest.top = eka2l1::vec2(
                (window_width - state->osd_w) / 2,
                (window_height - state->osd_h) / 2);
            osd_dest.size = eka2l1::vec2(state->osd_w, state->osd_h);

            eka2l1::rect osd_src;
            osd_src.size = eka2l1::vec2(state->osd_w, state->osd_h);

            builder.draw_bitmap(state->osd_texture, 0, osd_dest, osd_src);
            builder.set_feature(drivers::graphics_feature::blend, false);
        }

        builder.load_backup_state();

        state->present_status = -100;
        builder.present(&state->present_status);

        drivers::command_list retrieved = builder.retrieve_command_list();
        state->graphics_driver->submit_command_list(retrieved);
    }

    void emulator_state::register_draw_callback() {
        if (!winserv)
            return;

        epoc::screen *screens = winserv->get_screens();
        while (screens) {
            screens->add_screen_redraw_callback(this, draw_screen_callback);
            screens = screens->next;
        }
    }

    // Input helpers

    static drivers::input_event make_mouse_event_driver(float x, float y, float z, int button, int action, int mouse_id) {
        drivers::input_event evt;
        evt.type_ = drivers::input_event_type::touch;
        evt.mouse_.raw_screen_pos_ = false;
        evt.mouse_.pos_x_ = static_cast<int>(x);
        evt.mouse_.pos_y_ = static_cast<int>(y);
        evt.mouse_.pos_z_ = static_cast<int>(z);
        evt.mouse_.mouse_id = static_cast<std::uint32_t>(mouse_id);
        evt.mouse_.button_ = static_cast<drivers::mouse_button>(button);
        evt.mouse_.action_ = static_cast<drivers::mouse_action>(action);
        return evt;
    }

    static drivers::input_event make_key_event_driver(int key, drivers::key_state state) {
        drivers::input_event evt;
        evt.type_ = drivers::input_event_type::key;
        evt.key_.state_ = state;
        evt.key_.code_ = key;
        return evt;
    }

    void on_mouse_evt(void *userdata, vec3 pos, int button, int action, int mouse_id) {
        auto *emu = reinterpret_cast<emulator_state *>(userdata);
        const float scale = emu->symsys->get_config()->ui_scale;
        auto evt = make_mouse_event_driver(
            static_cast<float>(pos.x) / scale,
            static_cast<float>(pos.y) / scale,
            static_cast<float>(pos.z) / scale,
            button, action, mouse_id);

        const std::lock_guard<std::mutex> guard(emu->lockdown);
        if (emu->winserv)
            emu->winserv->queue_input_from_driver(evt);
    }

    static std::uint32_t remap_arrow_for_rotation(std::uint32_t key, int rotation) {
        if (rotation == 0)
            return key;

        // Map: physical key → game direction after clockwise rotation
        //  90°: Right→Up, Left→Down, Up→Left, Down→Right
        // 180°: Up→Down, Down→Up, Left→Right, Right→Left
        // 270°: Right→Down, Left→Up, Up→Right, Down→Left
        static const std::uint32_t table[3][4] = {
            // 90°:  Up→Left,    Down→Right,  Left→Down,   Right→Up
            { SDLK_LEFT, SDLK_RIGHT, SDLK_DOWN, SDLK_UP },
            // 180°: Up→Down,    Down→Up,     Left→Right,  Right→Left
            { SDLK_DOWN, SDLK_UP, SDLK_RIGHT, SDLK_LEFT },
            // 270°: Up→Right,   Down→Left,   Left→Up,     Right→Down
            { SDLK_RIGHT, SDLK_LEFT, SDLK_UP, SDLK_DOWN },
        };

        int idx = -1;
        switch (key) {
        case SDLK_UP:    idx = 0; break;
        case SDLK_DOWN:  idx = 1; break;
        case SDLK_LEFT:  idx = 2; break;
        case SDLK_RIGHT: idx = 3; break;
        default: return key;
        }

        return table[(rotation / 90) - 1][idx];
    }

    void on_key_press(void *userdata, std::uint32_t key) {
        auto *emu = reinterpret_cast<emulator_state *>(userdata);

        if (key == SDLK_r) {
            emu->host_rotation.store((emu->host_rotation.load() + 90) % 360);
            return;
        }

        if (key == SDLK_F5) {
            emu->show_osd_requested.store(true);
            return;
        }

        key = remap_arrow_for_rotation(key, emu->host_rotation.load());
        auto evt = make_key_event_driver(static_cast<int>(key), drivers::key_state::pressed);

        const std::lock_guard<std::mutex> guard(emu->lockdown);
        if (emu->winserv)
            emu->winserv->queue_input_from_driver(evt);
    }

    void on_key_release(void *userdata, std::uint32_t key) {
        auto *emu = reinterpret_cast<emulator_state *>(userdata);
        key = remap_arrow_for_rotation(key, emu->host_rotation.load());
        auto evt = make_key_event_driver(static_cast<int>(key), drivers::key_state::released);

        const std::lock_guard<std::mutex> guard(emu->lockdown);
        if (emu->winserv)
            emu->winserv->queue_input_from_driver(evt);
    }

    // Thread functions

    static void graphics_driver_thread(emulator_state &state) {
        common::set_thread_name("Graphics thread");
        common::set_thread_priority(common::thread_priority_high);

        auto wsi = state.window->get_window_system_info();

        bool use_gles = false;
#if defined(__unix__) && !defined(__ANDROID__)
#if defined(__aarch64__)
        use_gles = true;
#else
        use_gles = (wsi.type == drivers::window_system_type::wayland);
#endif
#endif

        auto sdl_gl_ctx = new gl_context_sdl2(state.window->get_sdl_window(), use_gles);
        wsi.external_gl_context = sdl_gl_ctx;

        state.graphics_driver = drivers::create_graphics_driver(drivers::graphic_api::opengl, wsi);
        state.graphics_driver->update_surface_size(state.window->window_fb_size());

        state.window->resize_hook = [](void *userdata, const vec2 &size) {
            auto *s = reinterpret_cast<emulator_state *>(userdata);
            s->graphics_driver->update_surface_size(size);
        };

        state.symsys->set_graphics_driver(state.graphics_driver.get());

        drivers::emu_window *win = state.window.get();
        state.graphics_driver->set_display_hook([win]() {
            win->swap_buffer();
        });

        state.graphics_event.set();
        state.graphics_driver->run();

        if (state.stage_two_inited)
            state.graphics_event.wait();

        state.graphics_driver.reset();
    }

    static void os_thread(emulator_state &state) {
        common::set_thread_name("Symbian OS thread");
        common::set_thread_priority(common::thread_priority_high);

        bool first_time = true;

        while (true) {
            if (state.should_emu_quit)
                break;

            const bool success = state.stage_two();
            state.init_event.set();

            if (first_time) {
                state.graphics_event.wait();
                first_time = false;
            }

            if (success || state.should_emu_quit)
                break;

            state.init_event.reset();
            state.init_event.wait();
        }

        state.register_draw_callback();

        while (!state.should_emu_quit) {
            state.symsys->loop();

            if (state.should_emu_pause && !state.should_emu_quit) {
                state.pause_event.wait();
                state.pause_event.reset();
            }
        }

        state.kill_event.wait();
        state.symsys.reset();
        state.graphics_event.set();
    }

    void kill_emulator(emulator_state &state) {
        state.should_emu_quit = true;
        state.should_emu_pause = false;
        state.pause_event.set();

        kernel_system *kern = state.symsys ? state.symsys->get_kernel_system() : nullptr;
        if (kern)
            kern->stop_cores_idling();

        if (state.graphics_driver)
            state.graphics_driver->abort();

        state.init_event.set();
        state.kill_event.set();
    }

    // CLI handlers

    static bool help_handler(common::arg_parser *parser, void *, std::string *) {
        std::cout << parser->get_help_string();
        return false;
    }

    static bool list_devices_handler(common::arg_parser *, void *userdata, std::string *) {
        auto *emu = reinterpret_cast<emulator_state *>(userdata);
        auto &devices = emu->symsys->get_device_manager()->get_devices();

        for (std::size_t i = 0; i < devices.size(); i++) {
            std::cout << i << " : " << devices[i].model << " (" << devices[i].firmware_code << ")" << std::endl;
        }
        return false;
    }

    static bool list_apps_handler(common::arg_parser *, void *userdata, std::string *err) {
        auto *emu = reinterpret_cast<emulator_state *>(userdata);
        kernel_system *kern = emu->symsys->get_kernel_system();

        applist_server *svr = kern ? reinterpret_cast<applist_server *>(
            kern->get_by_name<service::server>(
                get_app_list_server_name_by_epocver(kern->get_epoc_version()))) : nullptr;

        if (!svr) {
            *err = "Can't get app list server!";
            return false;
        }

        auto &regs = svr->get_registerations();
        for (std::size_t i = 0; i < regs.size(); i++) {
            std::string name = common::ucs2_to_utf8(regs[i].mandatory_info.long_caption.to_std_string(nullptr));
            std::cout << i << " : " << name << " (UID: 0x" << std::hex << regs[i].mandatory_info.uid << std::dec << ")" << std::endl;
        }

        return false;
    }

    static bool app_run_handler(common::arg_parser *parser, void *userdata, std::string *err) {
        const char *tok = parser->next_token();
        if (!tok) {
            *err = "No application specified";
            return false;
        }

        std::string tokstr = tok;
        const char *cmdline_peek = parser->peek_token();
        std::string cmdlinestr;

        if (cmdline_peek) {
            cmdlinestr = cmdline_peek;
            if (cmdlinestr.substr(0, 2) == "--") {
                cmdlinestr.clear();
            } else {
                parser->next_token();
            }
        }

        auto *emu = reinterpret_cast<emulator_state *>(userdata);
        kernel_system *kern = emu->symsys->get_kernel_system();

        applist_server *svr = kern ? reinterpret_cast<applist_server *>(
            kern->get_by_name<service::server>(
                get_app_list_server_name_by_epocver(kern->get_epoc_version()))) : nullptr;

        if (!svr) {
            *err = "Can't get app list server!";
            return false;
        }

        // UID-based launch
        if (tokstr.length() > 2 && tokstr.substr(0, 2) == "0x") {
            std::uint32_t uid = common::pystr(tokstr).as_int<std::uint32_t>();
            apa_app_registry *registry = svr->get_registration(uid);

            if (registry) {
                epoc::apa::command_line cmd;
                cmd.launch_cmd_ = epoc::apa::command_create;
                svr->launch_app(*registry, cmd, nullptr, [emu](kernel::process *) {
                    if (emu->app_started.load())
                        std::exit(0);
                });
                emu->app_launch_from_command_line = true;
                return true;
            }

            *err = "App with UID " + tokstr + " not found";
            return false;
        }

        // Path-based launch
        if (has_root_dir(tokstr)) {
            process_ptr pr = kern->spawn_new_process(common::utf8_to_ucs2(tokstr), common::utf8_to_ucs2(cmdlinestr));
            if (!pr) {
                *err = "Unable to launch process: " + tokstr;
                return false;
            }
            pr->logon([emu](kernel::process *) {
                if (emu->app_started.load())
                    std::exit(0);
            });
            pr->run();
            emu->app_launch_from_command_line = true;
            return true;
        }

        // Name-based launch
        auto &regs = svr->get_registerations();
        for (auto &reg : regs) {
            if (common::ucs2_to_utf8(reg.mandatory_info.long_caption.to_std_string(nullptr)) == tokstr) {
                epoc::apa::command_line cmd;
                cmd.launch_cmd_ = epoc::apa::command_create;
                svr->launch_app(reg, cmd, nullptr, [emu](kernel::process *) {
                    if (emu->app_started.load())
                        std::exit(0);
                });
                emu->app_launch_from_command_line = true;
                return true;
            }
        }

        *err = "No app found with name: " + tokstr;
        return false;
    }

    static bool device_set_handler(common::arg_parser *parser, void *userdata, std::string *err) {
        const char *device = parser->next_token();
        if (!device) {
            *err = "No device specified";
            return false;
        }

        auto *emu = reinterpret_cast<emulator_state *>(userdata);
        auto &devices = emu->symsys->get_device_manager()->get_devices();

        for (std::size_t i = 0; i < devices.size(); i++) {
            if (device == devices[i].firmware_code) {
                if (emu->conf.device != static_cast<int>(i)) {
                    emu->conf.device = static_cast<int>(i);
                    emu->symsys->set_device(static_cast<std::uint8_t>(i));
                }
                return true;
            }
        }

        *err = "Device not found: " + std::string(device);
        return false;
    }

    static bool install_handler(common::arg_parser *parser, void *userdata, std::string *err) {
        const char *path = parser->next_token();
        if (!path) {
            *err = "No SIS path given";
            return false;
        }

        auto *emu = reinterpret_cast<emulator_state *>(userdata);
        int result = emu->symsys->install_package(common::utf8_to_ucs2(path), drive_e);
        if (result != 0) {
            *err = "SIS installation failed (error " + std::to_string(result) + ")";
            return false;
        }
        std::cout << "SIS package installed successfully." << std::endl;
        return false;
    }

    static bool install_device_handler(common::arg_parser *parser, void *userdata, std::string *err) {
        const char *rpkg_path = parser->next_token();
        if (!rpkg_path) {
            *err = "Usage: --installdevice <RPKG_PATH> <ROM_PATH>";
            return false;
        }
        const char *rom_path = parser->next_token();
        if (!rom_path) {
            *err = "Usage: --installdevice <RPKG_PATH> <ROM_PATH>";
            return false;
        }

        auto *emu = reinterpret_cast<emulator_state *>(userdata);
        device_manager *dvcmngr = emu->symsys->get_device_manager();

        const std::string root_z_path = add_path(emu->conf.storage, "drives/z/");
        const std::string rom_resident_path = add_path(emu->conf.storage, "roms/");

        common::create_directories(root_z_path);
        common::create_directories(rom_resident_path);

        std::string firmware_code;

        auto progress_cb = [](const std::size_t taken, const std::size_t total) {
            int pct = static_cast<int>(taken * 100 / total);
            std::cout << "\rInstalling... " << pct << "%" << std::flush;
        };

        std::cout << "Installing RPKG: " << rpkg_path << std::endl;
        device_installation_error error = loader::install_rpkg(dvcmngr, rpkg_path, root_z_path,
            firmware_code, progress_cb, nullptr);
        std::cout << std::endl;

        if (error != device_installation_none) {
            *err = "RPKG installation failed (error " + std::to_string(static_cast<int>(error)) + ")";
            return false;
        }

        dvcmngr->save_devices();

        const std::string rom_dir = add_path(emu->conf.storage, add_path("roms", firmware_code + "/"));
        common::create_directories(rom_dir);
        const std::string rom_dest = add_path(rom_dir, "SYM.ROM");

        std::cout << "Copying ROM to " << rom_dest << std::endl;
        if (!common::copy_file(rom_path, rom_dest, true)) {
            *err = "Failed to copy ROM file";
            return false;
        }

        device *dvc = dvcmngr->lastest();
        if (dvc) {
            std::cout << "Device installed: " << dvc->model << " (" << dvc->firmware_code << ")" << std::endl;
        }
        std::cout << "Done! You can now run the emulator." << std::endl;
        return false;
    }

    static bool mount_card_handler(common::arg_parser *parser, void *userdata, std::string *err) {
        auto *emu = reinterpret_cast<emulator_state *>(userdata);
        const char *path = parser->next_token();
        if (!path) {
            *err = "No folder specified";
            return true;
        }

        io_system *io = emu->symsys->get_io_system();
        io->unmount(drive_e);

        if (common::is_dir(path)) {
            io->mount_physical_path(drive_e, drive_media::physical,
                io_attrib_removeable | io_attrib_write_protected,
                common::utf8_to_ucs2(path));
        } else {
            std::cout << "Mounting ZIP, please wait..." << std::endl;
            auto error = emu->symsys->mount_game_zip(drive_e, drive_media::physical, path, io_attrib_write_protected);
            if (error != zip_mount_error_none) {
                *err = "ZIP mount failed";
                return false;
            }
        }
        return true;
    }

    static TTF_Font *load_launcher_font(int size) {
        static const char *font_paths[] = {
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            nullptr
        };

        for (int i = 0; font_paths[i]; i++) {
            TTF_Font *font = TTF_OpenFont(font_paths[i], size);
            if (font)
                return font;
        }
        return nullptr;
    }

    static void draw_text(SDL_Renderer *renderer, TTF_Font *font,
            const std::string &text, int x, int y, SDL_Color color) {
        if (text.empty()) return;
        SDL_Surface *surface = TTF_RenderUTF8_Blended(font, text.c_str(), color);
        if (!surface) return;
        SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surface);
        SDL_Rect dst = { x, y, surface->w, surface->h };
        SDL_FreeSurface(surface);
        SDL_RenderCopy(renderer, tex, nullptr, &dst);
        SDL_DestroyTexture(tex);
    }

    static void draw_text_centered(SDL_Renderer *renderer, TTF_Font *font,
            const std::string &text, int center_x, int y, SDL_Color color) {
        if (text.empty()) return;
        SDL_Surface *surface = TTF_RenderUTF8_Blended(font, text.c_str(), color);
        if (!surface) return;
        SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surface);
        SDL_Rect dst = { center_x - surface->w / 2, y, surface->w, surface->h };
        SDL_FreeSurface(surface);
        SDL_RenderCopy(renderer, tex, nullptr, &dst);
        SDL_DestroyTexture(tex);
    }

    static SDL_Texture *load_app_icon(SDL_Renderer *renderer, emulator_state &state,
            applist_server *svr, fbs_server *fbss, apa_app_registry &reg, int icon_size) {
        io_system *io = state.symsys->get_io_system();
        std::string app_name = common::ucs2_to_utf8(reg.mandatory_info.long_caption.to_std_string(nullptr));

        const std::u16string path_ext = common::lowercase_ucs2_string(path_extension(reg.icon_file_path));
        std::string icon_path_utf8 = common::ucs2_to_utf8(reg.icon_file_path);

        LOG_TRACE(FRONTEND_CMDLINE, "Icon for '{}': path='{}' ext='{}' fbss={}",
            app_name, icon_path_utf8,
            common::ucs2_to_utf8(path_ext),
            fbss ? "yes" : "no");

        // Try MIF icon file (SVG-based vector icons)
        if (path_ext == u".mif" && io) {
            symfile file_route = io->open_file(reg.icon_file_path, READ_MODE | BIN_MODE);
            if (file_route) {
                ro_file_stream file_route_stream(file_route.get());
                loader::mif_file file_mif_parser(reinterpret_cast<common::ro_stream *>(&file_route_stream));

                if (file_mif_parser.do_parse()) {
                    std::vector<std::uint8_t> data;
                    int dest_size = 0;
                    if (file_mif_parser.read_mif_entry(0, nullptr, dest_size) && dest_size > 0) {
                        data.resize(dest_size);
                        file_mif_parser.read_mif_entry(0, data.data(), dest_size);

                        common::ro_buf_stream inside_stream(data.data(), data.size());
                        loader::mif_icon_header header;
                        inside_stream.read(&header, sizeof(loader::mif_icon_header));

                        if (header.type == loader::mif_icon_type_svg) {
                            std::string tmp_path = fmt::format("/tmp/eka2l1_icon_{:08X}.svg", reg.mandatory_info.uid);
                            {
                                common::wo_std_file_stream outfile(tmp_path, true);
                                std::vector<loader::svgb_convert_error_description> errors;
                                if (!loader::convert_svgb_to_svg(inside_stream, outfile, errors)) {
                                    if (!errors.empty() && errors[0].reason_ == loader::svgb_convert_error_invalid_file) {
                                        outfile.write(reinterpret_cast<const char *>(data.data()) + sizeof(loader::mif_icon_header),
                                            data.size() - sizeof(loader::mif_icon_header));
                                    }
                                }
                            }
                            auto document = lunasvg::Document::loadFromFile(tmp_path);
                            common::remove(tmp_path);

                            if (document) {
                                auto bitmap = document->renderToBitmap(icon_size, icon_size);
                                if (bitmap.valid()) {
                                    bitmap.convertToRGBA();
                                    SDL_Surface *surf = SDL_CreateRGBSurfaceFrom(
                                        bitmap.data(), bitmap.width(), bitmap.height(), 32, bitmap.stride(),
                                        0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
                                    if (surf) {
                                        SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surf);
                                        SDL_FreeSurface(surf);
                                        LOG_TRACE(FRONTEND_CMDLINE, "  MIF/SVG icon loaded OK");
                                        return tex;
                                    }
                                }
                            }
                        } else {
                            // Try NVG to SVG conversion
                            std::string tmp_path = fmt::format("/tmp/eka2l1_icon_{:08X}.svg", reg.mandatory_info.uid);
                            {
                                inside_stream = common::ro_buf_stream(
                                    data.data() + sizeof(loader::mif_icon_header),
                                    data.size() - sizeof(loader::mif_icon_header));
                                common::wo_std_file_stream outfile(tmp_path, true);
                                std::vector<loader::nvg_convert_error_description> errors_nvg;
                                loader::convert_nvg_to_svg(inside_stream, outfile, errors_nvg);
                            }
                            auto document = lunasvg::Document::loadFromFile(tmp_path);
                            common::remove(tmp_path);

                            if (document) {
                                auto bitmap = document->renderToBitmap(icon_size, icon_size);
                                if (bitmap.valid()) {
                                    bitmap.convertToRGBA();
                                    SDL_Surface *surf = SDL_CreateRGBSurfaceFrom(
                                        bitmap.data(), bitmap.width(), bitmap.height(), 32, bitmap.stride(),
                                        0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
                                    if (surf) {
                                        SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surf);
                                        SDL_FreeSurface(surf);
                                        LOG_TRACE(FRONTEND_CMDLINE, "  MIF/NVG icon loaded OK");
                                        return tex;
                                    }
                                }
                            }
                        }
                        LOG_TRACE(FRONTEND_CMDLINE, "  MIF icon conversion/render failed");
                    } else {
                        LOG_TRACE(FRONTEND_CMDLINE, "  MIF read_entry failed");
                    }
                } else {
                    LOG_TRACE(FRONTEND_CMDLINE, "  MIF parse failed");
                }
            } else {
                LOG_TRACE(FRONTEND_CMDLINE, "  MIF file open failed");
            }
        }

        // Try MBM icon file
        if (path_ext == u".mbm" && io) {
            symfile file_route = io->open_file(reg.icon_file_path, READ_MODE | BIN_MODE);
            if (file_route) {
                ro_file_stream file_route_stream(file_route.get());
                loader::mbm_file file_mbm_parser(reinterpret_cast<common::ro_stream *>(&file_route_stream));

                if (file_mbm_parser.do_read_headers() && !file_mbm_parser.sbm_headers.empty()) {
                    auto *hdr = &file_mbm_parser.sbm_headers[0];
                    int w = hdr->size_pixels.x, h = hdr->size_pixels.y;
                    LOG_TRACE(FRONTEND_CMDLINE, "  MBM icon: {}x{}", w, h);
                    std::vector<std::uint8_t> rgba(w * h * 4);
                    common::wo_buf_stream wstream(rgba.data(), rgba.size());

                    if (epoc::convert_to_rgba8888(fbss, file_mbm_parser, 0, wstream)) {
                        SDL_Surface *surf = SDL_CreateRGBSurfaceFrom(rgba.data(), w, h, 32, w * 4,
                            0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
                        if (surf) {
                            SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surf);
                            SDL_FreeSurface(surf);
                            LOG_TRACE(FRONTEND_CMDLINE, "  MBM icon loaded OK");
                            return tex;
                        }
                    } else {
                        LOG_TRACE(FRONTEND_CMDLINE, "  MBM convert_to_rgba8888 failed");
                    }
                } else {
                    LOG_TRACE(FRONTEND_CMDLINE, "  MBM read_headers failed");
                }
            } else {
                LOG_TRACE(FRONTEND_CMDLINE, "  MBM file open failed");
            }
        }

        // Try get_icon (works for apps with bitmap icons in the applist server)
        if (fbss) {
            std::optional<apa_app_masked_icon_bitmap> icon_pair = svr->get_icon(reg, 0);
            if (icon_pair.has_value() && icon_pair->first) {
                epoc::bitwise_bitmap *bmp = icon_pair->first;
                int w = bmp->header_.size_pixels.x, h = bmp->header_.size_pixels.y;
                LOG_TRACE(FRONTEND_CMDLINE, "  get_icon bitmap: {}x{}", w, h);
                if (w > 0 && h > 0 && w < 512 && h < 512) {
                    std::vector<std::uint8_t> rgba(w * h * 4);
                    common::wo_buf_stream wstream(rgba.data(), rgba.size());

                    if (epoc::convert_to_rgba8888(fbss, bmp, wstream)) {
                        SDL_Surface *surf = SDL_CreateRGBSurfaceFrom(rgba.data(), w, h, 32, w * 4,
                            0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
                        if (surf) {
                            SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surf);
                            SDL_FreeSurface(surf);
                            LOG_TRACE(FRONTEND_CMDLINE, "  get_icon loaded OK");
                            return tex;
                        }
                    } else {
                        LOG_TRACE(FRONTEND_CMDLINE, "  get_icon convert failed");
                    }
                }
            } else {
                LOG_TRACE(FRONTEND_CMDLINE, "  get_icon returned nothing");
            }
        }

        LOG_TRACE(FRONTEND_CMDLINE, "  No icon found for '{}'", app_name);
        return nullptr;
    }

    static SDL_Texture *make_placeholder_icon(SDL_Renderer *renderer, TTF_Font *font,
            const std::string &name, std::uint32_t uid, int size) {
        std::uint32_t hash = uid * 2654435761u;
        Uint8 r = 60 + (hash & 0x7F);
        Uint8 g = 60 + ((hash >> 8) & 0x7F);
        Uint8 b = 60 + ((hash >> 16) & 0x7F);

        SDL_Surface *surf = SDL_CreateRGBSurface(0, size, size, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000);
        if (!surf) return nullptr;

        SDL_FillRect(surf, nullptr, SDL_MapRGBA(surf->format, r, g, b, 255));

        // Draw border
        SDL_Rect top = { 0, 0, size, 1 }, bot = { 0, size - 1, size, 1 };
        SDL_Rect lft = { 0, 0, 1, size }, rgt = { size - 1, 0, 1, size };
        Uint32 border_col = SDL_MapRGBA(surf->format, r / 2, g / 2, b / 2, 255);
        SDL_FillRect(surf, &top, border_col);
        SDL_FillRect(surf, &bot, border_col);
        SDL_FillRect(surf, &lft, border_col);
        SDL_FillRect(surf, &rgt, border_col);

        // Draw first letter
        if (font && !name.empty()) {
            char letter[2] = { name[0], 0 };
            if (letter[0] >= 'a' && letter[0] <= 'z') letter[0] -= 32;
            SDL_Color white = { 255, 255, 255, 255 };
            SDL_Surface *txt = TTF_RenderUTF8_Blended(font, letter, white);
            if (txt) {
                SDL_Rect dst = { (size - txt->w) / 2, (size - txt->h) / 2, txt->w, txt->h };
                SDL_BlitSurface(txt, nullptr, surf, &dst);
                SDL_FreeSurface(txt);
            }
        }

        SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surf);
        SDL_FreeSurface(surf);
        return tex;
    }

    // Render OSD overlay content to an RGBA pixel buffer (CPU-side, no SDL_Renderer).
    // The graphics thread picks this up in draw_screen_impl and draws it as a GL texture.
    static void render_osd_pixels(emulator_state &state, TTF_Font *font,
            int menu_w, int menu_h, int pad, const char *rotate_name) {
        SDL_Surface *canvas = SDL_CreateRGBSurface(0, menu_w, menu_h, 32,
            0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
        if (!canvas) return;

        SDL_FillRect(canvas, nullptr, SDL_MapRGBA(canvas->format, 0, 0, 0, 180));

        Uint32 white_col = SDL_MapRGBA(canvas->format, 255, 255, 255, 255);
        SDL_Rect borders[] = {
            { 0, 0, menu_w, 2 },
            { 0, menu_h - 2, menu_w, 2 },
            { 0, 0, 2, menu_h },
            { menu_w - 2, 0, 2, menu_h }
        };
        for (auto &b : borders) SDL_FillRect(canvas, &b, white_col);

        SDL_Color white = { 255, 255, 255, 255 };
        SDL_Surface *label = TTF_RenderUTF8_Blended(font, "Rotate:", white);
        if (label) {
            SDL_SetSurfaceBlendMode(label, SDL_BLENDMODE_BLEND);
            SDL_Rect dst = { pad, pad, label->w, label->h };
            SDL_BlitSurface(label, nullptr, canvas, &dst);
            SDL_FreeSurface(label);
        }

        char val_buf[32];
        snprintf(val_buf, sizeof(val_buf), "<  %s  >", rotate_name);
        SDL_Color green = { 0, 255, 0, 255 };
        SDL_Surface *val = TTF_RenderUTF8_Blended(font, val_buf, green);
        if (val) {
            SDL_SetSurfaceBlendMode(val, SDL_BLENDMODE_BLEND);
            SDL_Rect dst = { menu_w - pad - val->w, pad, val->w, val->h };
            SDL_BlitSurface(val, nullptr, canvas, &dst);
            SDL_FreeSurface(val);
        }

        {
            const std::lock_guard<std::mutex> guard(state.osd_mutex);
            state.osd_w = menu_w;
            state.osd_h = menu_h;
            state.osd_pixels.resize(menu_w * menu_h * 4);
            std::memcpy(state.osd_pixels.data(), canvas->pixels, menu_w * menu_h * 4);
            state.osd_texture_dirty = true;
        }

        SDL_FreeSurface(canvas);
    }

    // OSD settings menu — renders to pixel buffer, drawn by the GL pipeline in draw_screen_impl.
    void show_osd_menu(emulator_state &state) {
        static const char *rotate_names[] = { "0", "90", "180", "270" };
        static const int rotate_values[] = { 0, 90, 180, 270 };
        static const int ROTATE_COUNT = 4;

        if (TTF_Init() != 0) return;

        SDL_Window *sdl_win = state.window->get_sdl_window();
        int win_w, win_h;
        SDL_GetWindowSize(sdl_win, &win_w, &win_h);

        int font_size = std::max(14, win_h / 25);
        TTF_Font *font = load_launcher_font(font_size);
        if (!font) { TTF_Quit(); return; }

        int line_h = TTF_FontLineSkip(font) * 3 / 2;
        int pad = font_size;
        int menu_w = std::min(win_w * 2 / 3, 500);
        int menu_h = line_h + pad * 2;

        if (state.osd_texture == 0 || state.osd_w != menu_w || state.osd_h != menu_h) {
            if (state.osd_texture != 0) {
                drivers::graphics_command_builder cleanup;
                cleanup.destroy_bitmap(state.osd_texture);
                auto cl = cleanup.retrieve_command_list();
                state.graphics_driver->submit_command_list(cl);
                state.osd_texture = 0;
            }
            state.osd_texture = drivers::create_bitmap(state.graphics_driver.get(),
                eka2l1::vec2(menu_w, menu_h), 32);
        }

        if (state.osd_texture == 0) {
            TTF_CloseFont(font);
            TTF_Quit();
            return;
        }

        int rot_idx = 0;
        int cur = state.host_rotation.load();
        for (int i = 0; i < ROTATE_COUNT; i++) {
            if (rotate_values[i] == cur) { rot_idx = i; break; }
        }

        render_osd_pixels(state, font, menu_w, menu_h, pad, rotate_names[rot_idx]);
        state.osd_visible.store(true);

        bool running = true;

        while (running) {
            if (process_termination_requested.load()) {
                state.should_emu_quit.store(true);
                break;
            }

            SDL_Event ev;
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_QUIT) {
                    state.should_emu_quit.store(true);
                    running = false;
                    break;
                }
                if (ev.type != SDL_KEYDOWN) continue;
                switch (ev.key.keysym.sym) {
                case SDLK_LEFT:
                    rot_idx = (rot_idx - 1 + ROTATE_COUNT) % ROTATE_COUNT;
                    state.host_rotation.store(rotate_values[rot_idx]);
                    render_osd_pixels(state, font, menu_w, menu_h, pad, rotate_names[rot_idx]);
                    break;
                case SDLK_RIGHT:
                    rot_idx = (rot_idx + 1) % ROTATE_COUNT;
                    state.host_rotation.store(rotate_values[rot_idx]);
                    render_osd_pixels(state, font, menu_w, menu_h, pad, rotate_names[rot_idx]);
                    break;
                case SDLK_F5:
                case SDLK_ESCAPE:
                case SDLK_RETURN:
                case SDLK_KP_ENTER:
                    running = false;
                    break;
                default:
                    break;
                }
            }

            SDL_Delay(16);
        }

        state.osd_visible.store(false);
        TTF_CloseFont(font);
        TTF_Quit();
    }

    bool show_app_launcher(emulator_state &state) {
        kernel_system *kern = state.symsys->get_kernel_system();
        if (!kern) return false;

        applist_server *svr = reinterpret_cast<applist_server *>(
            kern->get_by_name<service::server>(
                get_app_list_server_name_by_epocver(kern->get_epoc_version())));
        if (!svr) {
            LOG_ERROR(FRONTEND_CMDLINE, "App list server not available");
            return false;
        }

        fbs_server *fbss = reinterpret_cast<fbs_server *>(
            kern->get_by_name<service::server>(
                epoc::get_fbs_server_name_by_epocver(kern->get_epoc_version())));

        auto &regs = svr->get_registerations();
        if (regs.empty()) return false;

        struct app_entry {
            std::string name;
            std::uint32_t uid;
            int reg_index;
            SDL_Texture *icon = nullptr;
        };

        std::vector<app_entry> apps;
        for (std::size_t i = 0; i < regs.size(); i++) {
            if (regs[i].caps.is_hidden) continue;
            if ((regs[i].land_drive == drive_z) && (regs[i].mandatory_info.uid < 0x10300000)) continue;

            app_entry entry;
            entry.name = common::ucs2_to_utf8(regs[i].mandatory_info.long_caption.to_std_string(nullptr));
            entry.uid = regs[i].mandatory_info.uid;
            entry.reg_index = static_cast<int>(i);

            // Trim whitespace-only names
            bool has_visible = false;
            for (char c : entry.name) { if (c != ' ') { has_visible = true; break; } }
            if (!has_visible)
                entry.name = fmt::format("(0x{:08X})", entry.uid);

            apps.push_back(std::move(entry));
        }

        if (apps.empty()) return false;

        if (TTF_Init() != 0) {
            LOG_ERROR(FRONTEND_CMDLINE, "TTF_Init failed: {}", TTF_GetError());
            return false;
        }

        SDL_Window *win = SDL_CreateWindow("EKA2L1 - App Launcher",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            800, 600, SDL_WINDOW_SHOWN | SDL_WINDOW_FULLSCREEN_DESKTOP);
        if (!win) { TTF_Quit(); return false; }

        SDL_Renderer *renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
        if (!renderer) { SDL_DestroyWindow(win); TTF_Quit(); return false; }

        int win_w, win_h;
        SDL_GetWindowSize(win, &win_w, &win_h);

        const int icon_size = std::max(48, win_h / 8);
        const int cell_pad = 12;
        const int name_font_size = std::max(10, icon_size / 5);
        const int title_font_size = std::max(16, win_h / 30);

        TTF_Font *font_name = load_launcher_font(name_font_size);
        TTF_Font *font_title = load_launcher_font(title_font_size);
        TTF_Font *font_icon = load_launcher_font(icon_size / 2);

        if (!font_name) {
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(win);
            TTF_Quit();
            return false;
        }

        int name_line_h = TTF_FontLineSkip(font_name);
        int cell_w = icon_size + cell_pad * 2;
        int cell_h = icon_size + name_line_h * 2 + cell_pad * 2;
        int title_h = font_title ? TTF_FontLineSkip(font_title) + 16 : 40;
        int hint_h = name_line_h + 10;

        int grid_top = title_h;
        int grid_bottom = win_h - hint_h;
        int grid_area_h = grid_bottom - grid_top;

        int cols = std::max(1, win_w / cell_w);
        int rows_visible = std::max(1, grid_area_h / cell_h);

        // Center the grid horizontally
        int grid_left = (win_w - cols * cell_w) / 2;

        // Load icons
        LOG_INFO(FRONTEND_CMDLINE, "Loading icons for {} apps...", apps.size());
        for (auto &app : apps) {
            app.icon = load_app_icon(renderer, state, svr, fbss, regs[app.reg_index], icon_size);
            if (!app.icon)
                app.icon = make_placeholder_icon(renderer, font_icon, app.name, app.uid, icon_size);
        }
        LOG_INFO(FRONTEND_CMDLINE, "Icon loading complete");

        const SDL_Color color_bg = { 25, 25, 35, 255 };
        const SDL_Color color_title = { 100, 180, 255, 255 };
        const SDL_Color color_normal = { 190, 190, 190, 255 };
        const SDL_Color color_sel_text = { 255, 255, 255, 255 };
        const SDL_Color color_hint = { 128, 128, 128, 255 };
        const SDL_Color color_sel_bg = { 50, 80, 140, 255 };

        int selected = 0;
        int scroll_row = 0;
        bool running = true;
        bool app_selected = false;
        int total_count = static_cast<int>(apps.size());

        while (running) {
            if (process_termination_requested.load()) {
                state.should_emu_quit.store(true);
                running = false;
                break;
            }

            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                case SDL_QUIT:
                    state.should_emu_quit.store(true);
                    running = false;
                    break;
                case SDL_KEYDOWN:
                    switch (event.key.keysym.sym) {
                    case SDLK_RIGHT:
                        if (selected < total_count - 1) selected++;
                        break;
                    case SDLK_LEFT:
                        if (selected > 0) selected--;
                        break;
                    case SDLK_DOWN:
                        if (selected + cols < total_count) selected += cols;
                        else selected = total_count - 1;
                        break;
                    case SDLK_UP:
                        if (selected - cols >= 0) selected -= cols;
                        else selected = 0;
                        break;
                    case SDLK_PAGEDOWN:
                        selected = std::min(total_count - 1, selected + cols * rows_visible);
                        break;
                    case SDLK_PAGEUP:
                        selected = std::max(0, selected - cols * rows_visible);
                        break;
                    case SDLK_RETURN:
                    case SDLK_KP_ENTER: {
                        auto &app = apps[selected];
                        auto &reg = regs[app.reg_index];
                        epoc::apa::command_line cmd;
                        cmd.launch_cmd_ = epoc::apa::command_create;
                        state.app_exited.store(false);
                        svr->launch_app(reg, cmd, nullptr, [&state](kernel::process*) {
                            state.app_exited.store(true);
                        });
                        app_selected = true;
                        running = false;
                        break;
                    }
                    case SDLK_ESCAPE:
                        running = false;
                        break;
                    default:
                        break;
                    }
                    break;
                default:
                    break;
                }
            }

            // Keep selected item visible
            int sel_row = selected / cols;
            if (sel_row < scroll_row)
                scroll_row = sel_row;
            if (sel_row >= scroll_row + rows_visible)
                scroll_row = sel_row - rows_visible + 1;

            SDL_SetRenderDrawColor(renderer, color_bg.r, color_bg.g, color_bg.b, 255);
            SDL_RenderClear(renderer);

            // Title
            std::string title = "EKA2L1 - Select Application (" + std::to_string(total_count) + ")";
            draw_text_centered(renderer, font_title ? font_title : font_name, title,
                win_w / 2, 8, color_title);

            // Grid
            for (int row = scroll_row; row < scroll_row + rows_visible; row++) {
                for (int col = 0; col < cols; col++) {
                    int idx = row * cols + col;
                    if (idx >= total_count) break;

                    int cx = grid_left + col * cell_w + cell_pad;
                    int cy = grid_top + (row - scroll_row) * cell_h + cell_pad;
                    bool is_sel = (idx == selected);

                    if (is_sel) {
                        SDL_SetRenderDrawColor(renderer, color_sel_bg.r, color_sel_bg.g, color_sel_bg.b, 255);
                        SDL_Rect sel_rect = { cx - 4, cy - 4, cell_w - cell_pad + 8, cell_h - cell_pad + 8 };
                        SDL_RenderFillRect(renderer, &sel_rect);
                    }

                    // Icon
                    if (apps[idx].icon) {
                        SDL_Rect icon_rect = { cx, cy, icon_size, icon_size };
                        SDL_RenderCopy(renderer, apps[idx].icon, nullptr, &icon_rect);
                    }

                    // Name (truncate if too long)
                    std::string display_name = apps[idx].name;
                    int max_text_w = cell_w - cell_pad;
                    int text_w = 0, text_h = 0;
                    TTF_SizeUTF8(font_name, display_name.c_str(), &text_w, &text_h);
                    while (text_w > max_text_w && display_name.size() > 3) {
                        display_name.pop_back();
                        display_name.back() = '.';
                        TTF_SizeUTF8(font_name, display_name.c_str(), &text_w, &text_h);
                    }

                    int text_x = cx + (icon_size - text_w) / 2;
                    int text_y = cy + icon_size + 4;
                    draw_text(renderer, font_name, display_name, text_x, text_y,
                        is_sel ? color_sel_text : color_normal);
                }
            }

            // Scrollbar
            int total_rows = (total_count + cols - 1) / cols;
            if (total_rows > rows_visible) {
                int bar_x = win_w - 6;
                int bar_h = grid_area_h;
                int thumb_h = std::max(16, bar_h * rows_visible / total_rows);
                int thumb_y = grid_top + (bar_h - thumb_h) * scroll_row / std::max(1, total_rows - rows_visible);

                SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
                SDL_Rect bar_rect = { bar_x, grid_top, 4, bar_h };
                SDL_RenderFillRect(renderer, &bar_rect);

                SDL_SetRenderDrawColor(renderer, 120, 120, 120, 255);
                SDL_Rect thumb_rect = { bar_x, thumb_y, 4, thumb_h };
                SDL_RenderFillRect(renderer, &thumb_rect);
            }

            // Hints
            draw_text_centered(renderer, font_name,
                "Arrows: Navigate   Enter: Launch   Esc: Quit",
                win_w / 2, win_h - hint_h, color_hint);

            SDL_RenderPresent(renderer);
            SDL_Delay(16);
        }

        // Cleanup
        for (auto &app : apps) {
            if (app.icon) SDL_DestroyTexture(app.icon);
        }
        if (font_icon) TTF_CloseFont(font_icon);
        if (font_title && font_title != font_name) TTF_CloseFont(font_title);
        TTF_CloseFont(font_name);
        TTF_Quit();
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(win);

        return app_selected;
    }

}  // namespace eka2l1::sdl

int main(int argc, char *argv[]) {
    std::signal(SIGINT, eka2l1::sdl::handle_process_termination_signal);
    std::signal(SIGTERM, eka2l1::sdl::handle_process_termination_signal);

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) != 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    eka2l1::sdl::emulator_state state;
    state.stage_one();

    std::thread os_thread_obj(eka2l1::sdl::os_thread, std::ref(state));
    state.init_event.wait();

    eka2l1::common::arg_parser parser(argc, const_cast<const char **>(argv));

    parser.add("--help, -h", "Display help", eka2l1::sdl::help_handler);
    parser.add("--listapp", "List installed applications", eka2l1::sdl::list_apps_handler);
    parser.add("--listdevices", "List installed devices", eka2l1::sdl::list_devices_handler);
    parser.add("--app, -a, --run", "Run an app by name, UID (0x...), or virtual path", eka2l1::sdl::app_run_handler);
    parser.add("--device, -dvc", "Set device by firmware code", eka2l1::sdl::device_set_handler);
    parser.add("--install, -i", "Install a SIS package", eka2l1::sdl::install_handler);
    parser.add("--installdevice", "Install device from RPKG + ROM files", eka2l1::sdl::install_device_handler);
    parser.add("--mount, -m", "Mount a folder/zip as Game Card ROM on E:", eka2l1::sdl::mount_card_handler);

    if (argc > 1) {
        std::string err;
        state.should_emu_quit = !parser.parse(&state, &err);

        if (state.should_emu_quit) {
            state.graphics_event.set();
            state.kill_event.set();
            state.init_event.set();

            if (!err.empty())
                std::cerr << err << std::endl;

            os_thread_obj.join();
            SDL_Quit();
            return err.empty() ? 0 : 1;
        }
    }

    // Create the emulator window and graphics driver once; they persist across app launches
    // so that GPU resources (textures, bitmaps) remain valid.
    state.window = std::make_unique<eka2l1::sdl::emu_window_sdl2>();

    state.window->raw_mouse_event = eka2l1::sdl::on_mouse_evt;
    state.window->button_pressed = eka2l1::sdl::on_key_press;
    state.window->button_released = eka2l1::sdl::on_key_release;

    state.window->init("EKA2L1", eka2l1::vec2(800, 600), eka2l1::drivers::emu_window_flag_maximum_size);
    state.window->set_userdata(&state);
    state.window->close_hook = [](void *userdata) {
        auto *s = reinterpret_cast<eka2l1::sdl::emulator_state *>(userdata);
        s->should_emu_quit.store(true);
    };

    state.graphics_event.reset();
    std::thread graphics_thread_obj(eka2l1::sdl::graphics_driver_thread, std::ref(state));

    // Hide the emulator window until an app is actually running
    SDL_HideWindow(state.window->get_sdl_window());

    bool first_launch = true;

    while (true) {
        if (eka2l1::sdl::process_termination_requested.load()) {
            state.should_emu_quit.store(true);
            break;
        }

        if (!state.app_launch_from_command_line || !first_launch) {
            if (!eka2l1::sdl::show_app_launcher(state)) {
                break;
            }
        }
        first_launch = false;

        // Show the emulator window for the running app
        SDL_ShowWindow(state.window->get_sdl_window());
        SDL_RaiseWindow(state.window->get_sdl_window());

        state.app_exited.store(false);

        // Grace period: don't check window groups until the app has had time to open its window
        while (!state.should_emu_quit && !state.window->should_quit() && !state.app_exited.load()) {
            if (eka2l1::sdl::process_termination_requested.load()) {
                state.should_emu_quit.store(true);
                break;
            }

            state.window->poll_events();

            if (state.show_osd_requested.load()) {
                state.show_osd_requested.store(false);
                eka2l1::sdl::show_osd_menu(state);
            }

            SDL_Delay(1);
        }

        bool user_quit = state.should_emu_quit.load() || state.window->should_quit();
        if (user_quit)
            break;

        if (state.app_launch_from_command_line && state.app_exited.load()) {
            break;
        }

        // App exited: hide emulator window and go back to launcher
        SDL_HideWindow(state.window->get_sdl_window());
    }

    // Final cleanup: tear down graphics driver, OS thread, window
    eka2l1::sdl::kill_emulator(state);

    if (state.graphics_driver)
        state.graphics_driver->abort();
    state.graphics_event.set();
    graphics_thread_obj.join();

    state.window.reset();
    os_thread_obj.join();

    SDL_Quit();
    return 0;
}