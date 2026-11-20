mod model_hub;
mod mlx_runtime;
mod python_discovery;
mod report_pipeline;
mod settings_store;
mod stt_provisioning;
mod stt_worker;

use tauri::menu::MenuEvent;
#[cfg(not(target_os = "macos"))]
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager, RunEvent, WindowEvent};

use model_hub::ModelHubState;
use report_pipeline::AiRuntimeState;
use settings_store::SettingsStore;
use stt_worker::SttWorkerState;

const TRAY_SHOW: &str = "tray-show-session";
const TRAY_SETTINGS: &str = "tray-settings";
const TRAY_QUIT: &str = "tray-quit";

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let settings = SettingsStore::load(app.handle());
            app.manage(settings);
            app.manage(ModelHubState::default());
            app.manage(SttWorkerState::default());
            app.manage(AiRuntimeState::default());

            build_tray(app.handle())?;
            hide_main_on_close(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            settings_store::get_settings,
            settings_store::update_settings,
            settings_store::pick_directory,
            settings_store::pick_meeting_file,
            settings_store::import_legacy_dotenv,
            model_hub::model_hub_state,
            model_hub::model_hub_inspect,
            model_hub::model_hub_download,
            model_hub::model_hub_cancel,
            model_hub::model_hub_reset,
            model_hub::list_local_ai_models,
            model_hub::get_local_model_profile,
            model_hub::delete_local_ai_model,
            stt_provisioning::stt_model_status,
            stt_provisioning::install_selected_stt_model,
            stt_worker::start_session,
            stt_worker::stop_session,
            stt_worker::session_state,
            report_pipeline::run_report_on_file,
            report_pipeline::ai_runtime_status,
            report_pipeline::install_ai_runtime,
            report_pipeline::check_ai_runtime_update,
            report_pipeline::update_ai_runtime,
            report_pipeline::test_local_ai_model,
            report_pipeline::local_model_compatibility,
            report_pipeline::open_note,
            report_pipeline::open_settings_window,
            report_pipeline::open_session_window,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Meeting Transcriber");

    app.run(|app, event| match event {
        RunEvent::Exit => {
            app.state::<SttWorkerState>().shutdown();
            app.state::<AiRuntimeState>().shutdown();
        }
        #[cfg(target_os = "macos")]
        RunEvent::Reopen {
            has_visible_windows: false,
            ..
        } => {
            let _ = report_pipeline::show_window(app, report_pipeline::MAIN_WINDOW);
        }
        _ => {}
    });
}

/// Tray icon with the session, settings and quit entries.
#[cfg(target_os = "macos")]
fn build_tray(app: &AppHandle) -> tauri::Result<()> {
    // No `.menu(...)`: `tray-icon` would otherwise add a `TrayTarget` subview that forwards clicks
    // through `performClick`, which misaligns the menu. The `NSMenu` is attached to the status item
    // directly below, so `NSStatusBarButton` handles the clicks natively.
    let tray = TrayIconBuilder::with_id("main-tray")
        .tooltip("Meeting Transcriber")
        .icon(tauri::include_image!("icons/tray-template.png"))
        .icon_as_template(true)
        .on_menu_event(handle_tray_menu_event)
        .build(app)?;

    tray.with_inner_tray_icon(move |inner| {
        // `with_inner_tray_icon` runs on the main thread, so the muda menu is built here and never
        // needs to be `Send`.
        let show = muda::MenuItem::with_id(TRAY_SHOW, "Mostrar sesión", true, None);
        let settings = muda::MenuItem::with_id(TRAY_SETTINGS, "Configuración", true, None);
        let quit = muda::MenuItem::with_id(TRAY_QUIT, "Salir", true, None);
        let Ok(menu) = muda::Menu::with_items(&[&show, &settings, &quit]) else {
            return;
        };

        let Some(status_item) = inner.ns_status_item() else {
            return;
        };
        let Some(mtm) = objc2_foundation::MainThreadMarker::new() else {
            return;
        };
        if let Some(button) = status_item.button(mtm) {
            for subview in button.subviews() {
                subview.removeFromSuperview();
            }
        }

        let ns_menu_ptr = muda::ContextMenu::ns_menu(&menu).cast::<objc2_app_kit::NSMenu>();
        // SAFETY: `menu` is kept alive for the app lifetime below, so the `NSMenu` pointer stays valid.
        unsafe {
            if let Some(ns_menu) = ns_menu_ptr.as_ref() {
                status_item.setMenu(Some(ns_menu));
            }
        }

        // Keep the menu and its item targets alive so clicks on the native menu remain valid.
        std::mem::forget(menu);
    })?;
    Ok(())
}

/// Tray icon with the session, settings and quit entries.
#[cfg(not(target_os = "macos"))]
fn build_tray(app: &AppHandle) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, TRAY_SHOW, "Mostrar sesión", true, None::<&str>)?;
    let settings = MenuItem::with_id(app, TRAY_SETTINGS, "Configuración", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, TRAY_QUIT, "Salir", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &settings, &quit])?;

    TrayIconBuilder::with_id("main-tray")
        .tooltip("Meeting Transcriber")
        .menu(&menu)
        .icon(tauri::include_image!("icons/32x32.png"))
        .on_menu_event(handle_tray_menu_event)
        .build(app)?;
    Ok(())
}

fn handle_tray_menu_event(app: &AppHandle, event: MenuEvent) {
    match event.id().as_ref() {
        TRAY_SHOW => report_pipeline::show_main_tab(app, "session").unwrap_or_else(report_error),
        TRAY_SETTINGS => {
            report_pipeline::show_main_tab(app, "general").unwrap_or_else(report_error)
        }
        TRAY_QUIT => app.exit(0),
        _ => {}
    }
}

/// The main window is reused: closing it only hides it, so the app keeps running in the tray and
/// the next open is instant with its state intact.
fn hide_main_on_close(app: &AppHandle) {
    let Some(window) = app.get_webview_window(report_pipeline::MAIN_WINDOW) else {
        return;
    };
    let handle = window.clone();
    window.on_window_event(move |event| {
        if let WindowEvent::CloseRequested { api, .. } = event {
            api.prevent_close();
            let _ = handle.hide();
        }
    });
}

fn report_error(message: String) {
    eprintln!("tray: {message}");
}
